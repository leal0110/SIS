import cv2
import torch
import errno
import hashlib
import os
import sys
import tarfile
import numpy as np

import torch.utils.data as data
from PIL import Image
from six.moves import urllib
import json

from tqdm import tqdm

from mypath import Path


class COCOSegmentation(data.Dataset):

    category_names = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self,
                 root=Path.db_root_dir('pascal'),
                 split='val',
                 transform=None,
                 download=False,
                 preprocess=False,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 default=False):

        self._image_dir = 'PATH/TO/coco/train2014'
        self._mask_dir = 'PATH/TO/mask'

        self.transform = transform
        self.default = default

        self.obj_list = {}

        files = os.listdir(self._mask_dir)
        files.sort()
        print(self._mask_dir)
        print(len(files))

        for ii in tqdm(range(len(files))):
            self.obj_list[ii]  =  files[ii]

            assert os.path.isfile(os.path.join(self._image_dir, 'COCO_train2014_' + files[ii].split('_')[2] + ".jpg")) \
                , f'{files[ii]} IS NOT FOUND'

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'gt': _target}
        if self.transform is not None:
            sample = self.transform(sample)


        return sample

    def __len__(self):
        return len(self.obj_list)


    def _make_img_gt_point_pair(self, index):
        files = self.obj_list[index]
        image_id = 'COCO_train2014_' + files.split('_')[2]
        _img = np.array(Image.open(os.path.join(self._image_dir, image_id + ".jpg")).convert('RGB')).astype(np.float32)
        _tmp = np.array(Image.open(os.path.join(self._mask_dir, files))).astype(np.float32)

        if self.default:
            qwq = 0
        else:
            _target = (_tmp > (0.5*255)).astype(np.float32)
        return _img, _target

    def __str__(self):
        return 'coco pesudo'

