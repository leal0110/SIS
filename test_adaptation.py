from datetime import datetime

import imageio
from scipy.special import expit
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import socket
import timeit
from tqdm import tqdm

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloaders import custom_transforms as tr
from dataloaders.helpers import * 



from torch.nn.functional import interpolate as upsample
from networks.loss import *
from networks.correctnetwork import *
from networks.mas import *




def main():
    # Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU: {} '.format(gpu_id))

    # Setting parameters

    nEpochs = 1  # Number of epochs for training
    resume_epoch = 0  # Default is 0, change if want to resume
    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = 1  # Training batch size 5 
    snapshot = 5  # Store a model every snapshot epochs
    nInputChannels = 5  # Number of input channels (RGB + heatmap of extreme points)
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = 1e-5  # Learning rate
    p['wd'] = 0.0005  # Weight decay
    p['momentum'] = 0.9  # Momentum

    # correction
    threshold = 0.85  # loss
    # Results and model directories (a new directory is generated for every run)
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    if resume_epoch == 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    else:
        run_id = 0
    run_id = 0
    save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))

    # Network definition
    modelName = 'SIS_adaption'
    net = Network(nInputChannels=nInputChannels, num_classes=1,
                  backbone='resnet101',
                  output_stride=16,
                  sync_bn=None,
                  freeze_bn=False,
                  pretrained=False)
    if resume_epoch == 0:
        print("Initializing from pretrained model")
    else:
        print("Initializing weights from: {}".format(
            os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
        net.load_state_dict(
            torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                       map_location=lambda storage, loc: storage))

    train_params = [{'params': net.get_1x_lr_params(), 'lr': p['lr']},
                    {'params': net.get_10x_lr_params(), 'lr': p['lr']}]

    # init models with 5% pascal voc 2012 pretrained labels
    # bbox need onlybox= true
    net.load_state_dict(torch.load('SIS_refinement_epoch-99.pth',
                                   map_location=lambda storage, loc: storage))

    net.to(device)

    if resume_epoch != nEpochs:

        # Use the following optimizer
        optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
        p['optimizer'] = str(optimizer)

        # Preparation of the data loaders
        composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
            tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels'), relax=30, zero_pad=True),
            tr.FixedResize(
                resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},
                flagvals={'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                          'crop_void_pixels': cv2.INTER_LINEAR}),

            # 4 CHANNEL + 1 empty
            tr.IOGPoints(sigma=10, elem='crop_gt', pad_pixel=10, reverse_channel=True),

            tr.ToImage(norm_elem='IOG_points'),
            tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
            tr.ToTensor()])

        composed_transforms_ts = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels'), relax=30, zero_pad=True),
            tr.FixedResize(
                resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},
                flagvals={'gt': cv2.INTER_LINEAR, 'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                          'crop_void_pixels': cv2.INTER_LINEAR}),
            tr.IOGPoints(sigma=10, elem='crop_gt', pad_pixel=10, reverse_channel=True),

            tr.ToImage(norm_elem='IOG_points'),
            tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
            tr.ToTensor()])

        # you can compute omega during training phrase
        import dataloaders.subset_pascal as subset_pascal
        voc_train = subset_pascal.VOCSegmentation(split='05', transform=composed_transforms_tr)
        db_train = voc_train
        trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=2)

        net._init_reg_params_first_task()
        omega_optimizer = OmegaSgd(net.reg_params)
        net = compute_omega_grads_norm(net, trainloader, omega_optimizer)
        sanity_model(net)
        # lambda_reg gama
        optimizer = LocalSgd(
            params=train_params,
            lambda_reg=1.5,
            lr=p['lr'],
            momentum=p['momentum'],
            weight_decay=p['wd'])

        import dataloaders.ct as ct
        import dataloaders.mri as mri
        db_test = ct.Segmentation(transform=composed_transforms_ts, retname=True)
        # db_test = mri.Segmentation(transform=composed_transforms_ts, retname=True)

        img_num = len(db_test)
        print(img_num)
        p['dataset_train'] = str(db_test)
        p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]
        trainloader = DataLoader(db_test, batch_size=p['trainBatch'], shuffle=False, num_workers=2)


        # Train variables
        num_img_tr = len(trainloader)
        running_loss_tr = 0.0
        aveGrad = 0
        print("test adaption")

        
        # 5 CLICKS
        refinement_num_max = 3  # the number of new points:

        save_dir_res_list = []
        for add_clicks in range(0, refinement_num_max + 1):
            save_dir_res = os.path.join(save_dir, 'ResultsC-' + str(add_clicks))
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)
            save_dir_res_list.append(save_dir_res)

        net.train()
        net.freeze_bn()

        # default
        # with torch.no_grad():

        if True:
            for epoch in range(resume_epoch, nEpochs):

                for ii, sample_batched in enumerate(tqdm(trainloader)):
                    # print(sample_batched.keys())
                    optimizer.zero_grad()
                    crop_gts = sample_batched['crop_gt']
                    metas = sample_batched['meta']
                    gts = sample_batched['gt']
                    inputs = sample_batched['concat']
                    IOG_points = sample_batched['IOG_points']
                    for i in range(0, refinement_num_max + 1):

                        inputs.requires_grad_()
                        inputs, crop_gts, IOG_points = inputs.to(device), crop_gts.to(device), IOG_points.to(device)
                        if i == 0:
                            distance_map_512 = None
                            correction_map = -torch.ones(IOG_points.size(0), 1, IOG_points.size(2),
                                                             IOG_points.size(3)).cuda()
                        output_glo1, output_glo2, output_glo3, output_glo4, output_refine, iou_i, distance_map_512, \
                            correction_map = net.forward(inputs, IOG_points, crop_gts, i, distance_map_512, correction_map)

            

                        outputs = output_refine.to(torch.device('cpu'))
                        pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
                        pred = expit(pred)
                        pred = np.squeeze(pred)
                        gt = tens2image(gts[0, :, :, :])
                        bbox = get_bbox(gt, pad=30, zero_pad=True)
                        result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0, mask_relax=False)
                        result = (result * 255).astype(np.uint8)
                        imageio.imsave(
                            os.path.join(save_dir_res_list[i], metas['image'][0] + '-' + metas['object'][0] + '.png'),
                                result)

                        loss_c_refine = correction_cross_entropy_loss(output_refine, correction_map)
                        loss_c_refine = correction_cross_entropy_loss(output_refine, correction_map)
                        loss = loss_c_refine

                        # test
                        # if iou_i > threshold:
                        #         break
                        loss.backward()
                        optimizer.step(net.reg_params)


if __name__ == '__main__':
    main()
