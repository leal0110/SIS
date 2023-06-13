import torch
import os
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import imgviz
from cluster_utils import *
# import matplotlib.pyplot as plt

def compute_labels(args, centroids, dataloader=None, model=None, savepath=None):
    """
    Label all images with the obtained cluster centroids. 
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    # K = centroids.size(0)
    assert model != None or savepath != None, 'At least one way to get featuremaps!'

    if savepath:
        images = os.listdir(savepath)
        model = None
        
    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids)
    assert model is not None
    assert args.batch_size == 1, 'assign mode only support batch size 1'
    model.eval()

    area = torch.zeros([args.k]).long().cuda()

    with torch.no_grad():
        for images, labels, superpixel, name, ori_size in tqdm(dataloader):
            # if idx < 41370 :
            #     continue
            image = images.cuda(non_blocking=True)
            # print('image',image.shape)
            feats = model(image).detach()
            # print('feats', feats.shape)
            # Normalize.
            feats = F.normalize(feats, p=2, dim=1)
            feats = F.interpolate(feats,
                            size=superpixel.shape[-2:],
                            mode='bilinear',
                            align_corners=True)
            labels = labels.long()
            
            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(feats, centroids, metric_function) 
            
            # when batch size = 1
            gt = labels[0]
            scores, label = scores[0].max(dim=0)
            spl = superpixel.squeeze().long()

            # compute superpixel max pool
            spl_num = spl.max()
            # calculate vote cluster labels
            sp_label = torch.zeros_like(label)
            # flag = False
            for i in range(spl_num):
                if label[spl==i].shape[0] == 0:
                    continue
                # if torch.nonzero(spl==6).sum() > 32*32:
                #     flag = True
                votes = torch.bincount(label[spl==i])
                # print('votes', votes.shape, votes)
                l = votes.argmax()
                sp_label[spl==i] = l

            # sp_label[gt==255] = 255

            # save labels
            os.makedirs(args.label_root +'_super', exist_ok=True)
            # save_label(label, os.path.join(args.label_root, image_name +'.png'))

            for kk in range(len(torch.bincount(sp_label.flatten()))):
                area[kk] += torch.bincount(sp_label.flatten())[kk]

            save_label(sp_label, os.path.join(args.label_root +'_super', name[0] +'.png'))
        # area count
        np.save(f'area_{args.k}.npy', area.cpu().numpy())
        
     
def save_label(label, path):
    label = label.cpu().numpy()
    dst = Image.fromarray(label.astype(np.uint8), 'P')
    colormap = imgviz.label_colormap()
    dst.putpalette(colormap.flatten())
    # dst.putpalette()
    dst.save(path)
