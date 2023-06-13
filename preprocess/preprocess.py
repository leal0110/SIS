import argparse
import os
import imageio
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pascal', choices=['coco', 'pascal'], help='dataset to cluseter')
    parser.add_argument('--data_root', type=str, default='', help='dataset root')
    parser.add_argument('--label_root', type=str, default='./label')
    parser.add_argument('--k', type=int, default=30, help='K')
    return parser.parse_args()

def main():

    args = parse_arguments()

    args.label_root = "path/label"

    data_list = './coco_train.txt'
    list_read = open(data_list).readlines()

    sampled_img_list = []
    name_list = []

    # area = np.load(f'area_{args.k}.npy')
    # ex = np.argsort(area)[-int(args.k/6):]

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        file_name = line_split[0][:-4].split('/')[-1]
        name_list.append(file_name)
        _mask = np.array(Image.open(os.path.join(args.label_root, file_name + '.png')))
        for i in np.unique(_mask):
            if i not in ex:
                _target = _mask == i
                target_area = np.sum(_target)
                if target_area > np.prod(_mask.shape) * 0.01:
                    # Saliency Optimization from Robust Background Detection
                    # band = lenbnd(p) / area(p)
                    bnd = np.zeros_like(_target)
                    bnd[:, 0] = 1
                    bnd[:, -1] = 1
                    bnd[0, :] = 1
                    bnd[-1, :] = 1
                    lenbnd = np.sum(np.logical_and(_target, bnd))
                    band = lenbnd / pow(target_area, 0.5)
                    if band < 1:
                    # connectedComponents
                        num_objects, labels = cv2.connectedComponents(_target.astype(np.uint8))
                        for il in range(1, len(np.unique(labels))):
                            _obj = labels == il
                            obj_area = np.sum(_obj)
                            if obj_area > np.prod(_target.shape) * 0.01 or len(np.unique(labels)) == 2:
                                imageio.imsave(os.path.join('dino_30_001', file_name + '_' + str(i) + '_' + str(il) + '.png'), (_obj * 255).astype(np.uint8))

if __name__ == '__main__':
    main()
