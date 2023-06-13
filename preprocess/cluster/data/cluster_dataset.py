
import os
import cv2
import random
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from data import transformsp as transform
from skimage import segmentation

def my_collate(batch):
    print('batch', len(batch))
    print(batch[0][0].shape)
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def get_img_label(image_path=None, label_path=None):
    """
    :param image_path: str / None
    :param label_path: str
    :return:
        image: np float array of shape (H, W, 3)
        label: np int array of (H, W)
        unique_class: int list, classes exists in the image
    """
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if image_path is not None:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
    else:
        image = None

    return image, label

 
class CLusterData(Dataset):
    def __init__(self, mode, data_root, data_list, dataset='pascal', use_spl=False, sample_num=2000000):
        if not os.path.isfile(data_list):
            raise RuntimeError(f"Image list file do not exist: {data_list}\n")

        self.mode = mode

        list_read = open(data_list).readlines()

        print(len(list_read))
        if sample_num < len(list_read):
            sampled_idx = random.sample(range(len(list_read)), sample_num)
            remain_idx = list(set(range(len(list_read))) - set(sampled_idx))
        else:
            sampled_idx = range(len(list_read))

        sampled_img_list = []
        name_list = []

        # 5000 images
        # _size = 5000
        # if dataset == 'coco' and self.mode == 'cluster':
        #     np.random.seed(seed=2022)
        #     labeled_idx = np.random.choice(range(len(list_read)), size=_size, replace=False)
        #     sampled_idx = np.sort(labeled_idx)


        for l_idx in sampled_idx:

            # # for make up
            # line = list_read[l_idx]
            # line = line.strip()
            # image_name = os.path.join(data_root, 'train2014/' + line + '.jpg')
            # print(image_name)
            # label_name = os.path.join(data_root, 'annotations/train2014/' + line + '.png')
            # print(label_name)

            line = list_read[l_idx]
            line = line.strip()
            line_split = line.split(' ')

            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
            file_name = image_name[:-4].split('/')[-1]
            sampled_img_list.append(
                (image_name, label_name)
            )
            name_list.append(file_name)

        if sample_num < len(list_read):
            with open('./reamin.txt', 'w+') as f:
                for idx in remain_idx:
                    f.write(list_read[idx])

        self.img_list = sampled_img_list

        self.dataset = dataset
        self.name_list = name_list
        self.use_spl = use_spl
        self.superpixel_type = 'felzenszwalb'

        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        # transform_list = [
        #         transform.RandScale([0.9, 1.1]),
        #         transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
        #         transform.RandomGaussianBlur(),
        #         transform.RandomHorizontalFlip(),
        #         transform.Crop(473, crop_type='rand', padding=mean, ignore_label=255)
        #     ]
        # transform_list = [
        #     transform.ToTensor(),
        #     transform.Normalize(mean=mean, std=std),
        #     # transform.Resize(size=473)
        #     ]

        # self.transform = transform.MultiScaleNorm(scale=[1, 2], mean=mean, std=std)
        if self.mode == 'assign':
            self.transform = transform.Compose([],mean,std)
        else:
            self.transform = transform.Compose([transform.Resize(size=473)],mean,std)
 
    def __len__(self):
 
        return len(self.img_list)
 
    def __getitem__(self, index):
        image_path, label_path = self.img_list[index]
        name = self.name_list[index]
        image, label = get_img_label(image_path=image_path, label_path=label_path)
        ori_size = image.shape[:2]
        # transform

        if self.use_spl:
            superpixel = get_superpixel(image_path=image_path,
                                    method=self.superpixel_type)
            image, label, [superpixel] = self.transform(image, label, [superpixel])
            return image, label, superpixel, name, ori_size
        else:
            superpixel = None
            image, label, _ = self.transform(image, label)
            return image, label, name, ori_size


def get_superpixel(image_path, method):
    """
    Generate superpixel label
    255 -> ignore label
    1 to n -> the n base classes in the image
    0 is reserved for novel class
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if method == 'slic':
        superpixel = segmentation.slic(image, compactness=10, n_segments=100)  # 250
    elif method == 'felzenszwalb':
        superpixel = segmentation.felzenszwalb(image, scale=100, sigma=0.8, min_size=200)
    elif method == 'hed':
        image_name = image_path.split('/')[-1].split('.')[0]
        superpixel = cv2.imread(f'./hed/{image_name}.png', cv2.IMREAD_GRAYSCALE)
        superpixel = np.asarray(superpixel)
    else:
        raise ValueError(f'Do not recognise superpixel method {method}')


    return superpixel

def get_data_loader(args, dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

