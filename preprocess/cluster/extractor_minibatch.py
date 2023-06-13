# -*- coding: utf-8 -*-
"""
Function: 图像特征的提取、聚类、和聚类伪标签的生成。
Writer: sky
date: 2022.3.18
"""
from __future__ import print_function, division, absolute_import
from unittest.mock import patch
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import argparse
from torchvision import datasets
import os
import cv2
import time
import copy
import random
from tqdm import tqdm
# import torch.utils.data as data
# from data import transform
import torch.nn.functional as F
from model.extractor_model.resnet import resnet50_
from model.extractor_model.deeplab import deeplabv3_resnet50_
from model.extractor_model.vit import vit_base
from model.extractor_model.fpn import PanopticFPN
from data.cluster_dataset import CLusterData, get_data_loader
from assign_labels import *
from cluster_utils import *



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pascal', choices=['coco', 'pascal'], help='dataset to cluseter')
    parser.add_argument('--data_root', type=str, default='', help='dataset root')
    parser.add_argument('--label_root', type=str, default='./label')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--arch', type=str, default='moco', 
                        choices=['moco', 'deeplabv3', 'resnet50', 'dino', 'dino_resnet50'],
                        help='feature extractor architecture')
    parser.add_argument('--mode', type=str, default='cluster', choices=['cluster', 'assign', 'subcluster'])
    parser.add_argument('--in_dim', type=int, default=2048)
    # parser.add_argument('--use_pca', action='store_true', default=False, help='use pca to reduce feature dimension')
    parser.add_argument('--pca_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=30, help='聚类数')
    parser.add_argument('--niter', type=int, default=30, help='kmeans n iters')
    parser.add_argument('--ngpu', type=int, default=2, help='GPU个数')
    return parser.parse_args()

def feature_flatten(feats):
    if len(feats.size()) == 2:
        # feature already flattened. 
        return feats
    
    feats = feats.view(feats.size(0), feats.size(1), -1).transpose(2, 1)\
            .contiguous().view(-1, feats.size(1))
    
    return feats 

def run_mini_batch_kmeans(args, model, dataloader):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update. 
    """
    kmeans_loss  = AverageMeter()
    faiss_module = get_faiss_module(args)
    data_count   = np.zeros(args.k)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    args.seed = 2021
    args.num_init_batches = 100  # set according to your cuda memory
    args.num_init_batches = 157  # set according to your cuda memory
    args.num_init_batches = 300

    # test for all

    model.eval()

    print(len(dataloader))

    with torch.no_grad():
        for i_batch, batch in enumerate(tqdm(dataloader)):
            images, labels, name, ori_size = batch
            # 1. Compute initial centroids from the first few batches. 
            if isinstance(images, list):
                mlvs = len(images)
                mlv_fmap = []
                for i in range(mlvs):
                    image = images[i].cuda(non_blocking=True)
                    # label = labels[i].to(device)
                    feature = model(image)
                    scaled_featuer = F.interpolate(feature, size=(int(ori_size[0]/8), int(ori_size[1]/8)), mode='bilinear', align_corners=True)
                    mlv_fmap.append(scaled_featuer)
                fuse_feature = torch.cat(mlv_fmap, dim=1)
            else:
                images = images.cuda(non_blocking=True)
                fuse_feature = model(images).detach()

            # Normalize.
            # print('feats', fuse_feature.shape)
            fuse_feature = F.normalize(fuse_feature, p=2, dim=1)
            # feats = fuse_feature.squeeze().flatten(1)
            # feats = feats.permute(1,0)
            if i_batch == 0:
                print('Batch input size : {}'.format(list(images.shape)))
                print('Batch feature : {}'.format(list(fuse_feature.shape)))
            
            feats = feature_flatten(fuse_feature).detach().cpu()
            # print('feats', feats.shape, feats.unique())     


            if num_batches < args.num_init_batches:
                featslist.append(feats)
                num_batches += 1
                
                # if num_batches == args.num_init_batches or num_batches == len(dataloader):
                if num_batches == args.num_init_batches or i_batch == len(dataloader) - 1:
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).numpy().astype('float32')
                        print('featslist', featslist.shape)
                        centroids = get_init_centroids(args, args.k, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)

                        kmeans_loss.update(D.mean())
                        print('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False

                    else:
                        
                        b_feat = torch.cat(featslist).cpu().numpy().astype('float32')
                        print('b_feat', b_feat.shape)
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat, 1)

                        kmeans_loss.update(D.mean())
                        print('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        # Update centroids. 
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr    = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k]   = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(0).astype('float32')
                    
                    # Empty. ?
                    featslist   = []
                    num_batches = 0

            if (i_batch % args.num_init_batches) == 0:
                print('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))

    centroids = torch.tensor(centroids, requires_grad=False).cuda()

    return centroids, kmeans_loss.avg

def main():
    args = parse_arguments()
    if args.dataset == 'pascal':
        data_root = os.path.join(args.data_root, 'pascal')
    elif args.dataset == 'coco':
        data_root = os.path.join(args.data_root, 'COCO2014')
    data_list = {x: './lists/'+args.dataset+'/{}.txt'.format(x) for x in ['train', 'val', 'test']}
    data_list = {'train' : "./coco_train.txt"}
    # save_path = os.path.join(args.save_root, args.dataset+'_'+args.arch)
    args.label_root = os.path.join(args.label_root, args.dataset+'_'+args.arch)

    # build dataloader
    # load dataset
    dataset = CLusterData(args.mode, data_root, data_list['train'], args.dataset, use_spl=False)
    # print('train_datasete_size:', len(dataset))
    dataloader = get_data_loader(args,dataset)

    # chose pre-trained model
    if args.arch =='moco':
        pthpath = './weights/moco_v2_800ep_pretrain.pth.tar'
        model_ft = resnet50_(pretrained=False)
        ckpt = torch.load(pthpath)['state_dict']
        # print('state_dict')
        # for k, v in ckpt.items():
        #     print(k)
        format_dict = {k.replace('module.encoder_q.', ''): v  for k, v in ckpt.items() if not 'fc' in k}
        model_ft.load_state_dict(format_dict)
    elif args.arch =='deeplabv3':
        pthpath = './weights/best_deeplabv3_resnet50_voc_os16.pth'
        model_ft = deeplabv3_resnet50_(pretrained_backbone=False)
        ckpt = torch.load(pthpath)['model_state']
        model_ft.load_state_dict(ckpt)
    elif args.arch =='resnet50':
        pthpath = './weights/resnet50-19c8e357.pth'
        model_ft = resnet50_(pretrained=False)
        ckpt = torch.load(pthpath)
        print('state_dict')
        for k, v in ckpt.items():
            print(k)
        format_dict = {k: v  for k, v in ckpt.items() if not 'fc' in k}
        model_ft.load_state_dict(format_dict)
    elif args.arch == 'dino':
        pthpath = './weights/dino_vit_B_8.pth'
        model_ft = vit_base(patch_size=8)
        ckpt = torch.load(pthpath)
        # print('state_dict')
        # for k, v in ckpt.items():
        #     print(k)
        model_ft.load_state_dict(ckpt)
    elif args.arch == 'dino_resnet50':
        # model_ft = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        # ft shape 1,2048
        dino_resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model_ft = resnet50_(pretrained=False)
        # for k, v in dino_resnet50.state_dict().items():
        #     print(k)
        format_dict = {k: v for k, v in dino_resnet50.state_dict().items() if not 'head' in k}
        model_ft.load_state_dict(format_dict)

    elif args.arch == 'picie':
        pthpath = './weights/picie.pkl'
        model_ft = PanopticFPN('resnet18', pretrained=False)
        ckpt = torch.load(pthpath)['state_dict']
        # print('state_dict')
        # for k, v in ckpt.items():
        #     print(k)
        format_dict = {k.replace('module.', ''): v  for k, v in ckpt.items()}
        model_ft.load_state_dict(format_dict)
    
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.cuda()

    if not os.path.exists('centroids'):
            # os.makedir('centroids')
            os.mkdir('centroids')
    centroids_path = os.path.join('centroids', 'centroids_{}_{}_{}.npy'.format(args.arch, args.dataset, args.k))

    if args.mode == 'cluster':
        # K-means Cluster
        print("run")
        t0 = time.time()
        centroids, obj = run_mini_batch_kmeans(args, model_ft, dataloader)
        print("final objective: %.4g" % obj)
        np.save(centroids_path, centroids.cpu().numpy())
        t1 = time.time()
        print("total runtime: %.3f s" % (t1 - t0))

    if args.mode == 'assign':
        # # Assign labels per image
        args.batch_size = 1

        dataset_sup = CLusterData(args.mode, data_root, data_list['train'], args.dataset, use_spl=True)
        dataloader_sup = get_data_loader(args,dataset_sup)

        centroids = np.load(centroids_path)
        centroids = torch.from_numpy(centroids).cuda()

        compute_labels(args, centroids, dataloader=dataloader_sup, model=model_ft)

 
if __name__ == '__main__':
    main()
