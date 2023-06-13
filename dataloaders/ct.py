import random
import numpy as np
import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import json

class Segmentation(data.Dataset):
    def __init__(self,
                 split='val',
                 transform=None,
                 preprocess=False,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 default=False):

        _root = 'PATH/TO/CT'
        _mask_dir = 'PATH/TO/CT/label'  # each object each color
        _image_dir = 'PATH/TO/CT/img'
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels
        self.default = default

        # Build the ids file
        area_th_str = ""
        if self.area_thres != 0:
            area_th_str = '_area_thres-' + str(area_thres)

        self.obj_list_file = os.path.join(_root, '_'.join(self.split) + '_instances' + area_th_str + '.txt')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_root, 'list')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.masks = []
        manual_seed = 42
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        xx = []
        for did in range(2,49):
            if did in [15,23,36,37]:
                continue
            filepath = r'PATH/TO/CT/label/' + str(did).zfill(3)
            filenames = os.listdir(filepath)
            filenames.sort()
            f_list = []
            for f in filenames:
                if (len(f[:-6]) == 1):
                    f_list.append('00' + f[:-6])
                elif (len(f[:-6]) == 2):
                    f_list.append('0' + f[:-6])
                else:
                    f_list.append(f[:-6])
            labeled_idx = np.random.choice(range(len(f_list)), size=3, replace=False)

            for id in labeled_idx:
                xx.append([str(did).zfill(3), f_list[id]])
        xx.sort()

        for splt in self.split:
            for i in range(len(xx)):
                _image = os.path.join(_image_dir, xx[i][0]+'\\' + '1-'+ xx[i][1] + '.png')
                _mask = os.path.join(_mask_dir, xx[i][0]+'\\' + str(int(xx[i][1])) + "_1.png")

                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.im_ids.append(xx[i][0]+'_'+ xx[i][1])
                self.images.append(_image)
                self.masks.append(_mask)
        assert (len(self.images) == len(self.masks))

        # Precompute the list of objects and their categories for each image
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing of sstem dataset, this will take long, but it will be done only once.')
            self._preprocess()

        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            flag = False
            for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                if self.obj_dict[self.im_ids[ii]][jj] != -1:
                    self.obj_list.append([ii, jj])
                    flag = True
            if flag:
                num_images += 1

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        _img, _target, _void_pixels, _, _, _ = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'category': self.obj_dict[self.im_ids[_im_ii]][_obj_ii],
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.obj_list)

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]
        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32)
        _tmp = (np.array(Image.open(self.masks[_im_ii]))).astype(np.float32)
        _void_pixels = (_tmp == -1)
        _other_same_class = np.zeros(_tmp.shape)
        _other_classes = np.zeros(_tmp.shape)

        if self.default:
            _target = _tmp
            _background = np.logical_and(_tmp == 0, ~_void_pixels)
        else:
            _target = (_tmp  > 0.5*255 ).astype(np.float32)
            _background = np.logical_and(_tmp == 0, ~_void_pixels)

        return _img, _target, _void_pixels.astype(np.float32), \
               _other_classes.astype(np.float32), _other_same_class.astype(np.float32), \
               _background.astype(np.float32)

    def __str__(self):
        return 'ct test dataset'

    def _check_preprocess(self):
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))

            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        self.obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _mask = np.array(Image.open(self.masks[ii]))
            _mask_ids = np.unique(_mask)
            _cat_ids = [1]
            self.obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

if __name__ == '__main__':

    import dataloaders.custom_transforms as tr
    from torchvision import transforms
    transform = transforms.Compose([tr.ToTensor()])
    dataset = Segmentation(split=['val'], transform=transform, retname=True)