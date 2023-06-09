import numpy.random as random
import dataloaders.helpers as helpers
from dataloaders.helpers import *
import torch.nn.functional as F
import torchvision.transforms as TF


class RandomVerticalTensorFlip(object):
    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.plist = np.random.random_sample(N)

    def __call__(self, indice, image):
        I = np.nonzero(self.plist[indice] < self.p_ref)[0]

        if len(image.size()) == 3:
            image_t = image[I].flip([1])
        else:
            image_t = image[I].flip([2])

        return torch.stack([image_t[np.where(I == i)[0][0]] if i in I else image[i] for i in range(image.size(0))])


#
#
class RandomHorizontalTensorFlip(object):
    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.plist = np.random.random_sample(N)

    def __call__(self, indice, image, is_label=False):
        I = np.nonzero(self.plist[indice] < self.p_ref)[0]

        if len(image.size()) == 3:
            image_t = image[I].flip([2])
        else:
            image_t = image[I].flip([3])

        return torch.stack([image_t[np.where(I == i)[0][0]] if i in I else image[i] for i in range(image.size(0))])


#
#
class RandomResizedCrop(object):
    def __init__(self, N, res, scale=(0.5, 1.0)):
        self.res = res
        self.scale = scale
        self.rscale = [np.random.uniform(*scale) for _ in range(N)]
        self.rcrop = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(N)]

    def random_crop(self, idx, img):
        ws, hs = self.rcrop[idx]
        res1 = int(img.size(-1))
        res2 = int(self.rscale[idx] * res1)
        i1 = int(round((res1 - res2) * ws))
        j1 = int(round((res1 - res2) * hs))

        return img[:, :, i1:i1 + res2, j1:j1 + res2]

    def __call__(self, indice, image):
        new_image = []
        res_tar = self.res // 4 if image.size(1) > 5 else self.res  # View 1 or View 2?

        for i, idx in enumerate(indice):
            img = image[[i]]
            img = self.random_crop(idx, img)
            img = F.interpolate(img, res_tar, mode='bilinear', align_corners=False)

            new_image.append(img)

        new_image = torch.cat(new_image)

        return new_image


class RandomRotate(object):
    def __init__(self, N, p=0.5, rots=(-20, 20)):
        self.plist = np.random.random_sample(N)
        self.rots = rots

    def __call__(self, indice, image, fill=0):
        img = []
        for ii in range(image.size(0)):
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * self.plist[indice][ii] - \
                  (self.rots[1] - self.rots[0]) / 2
            rr = TF.RandomRotation(degrees=(rot, rot), fill=fill)
            image_t = rr(image[ii])
            img.append(image_t)

        return torch.stack(img)


#

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """

    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0]) / 2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert (center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot=' + str(self.rots) + ',scale=' + str(self.scales) + ')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """

    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert (len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())

        for elem in elems:

            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3] - bbox[1] + 1, bbox[4] - bbox[2] + 1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[elem] = np.round(sample[elem] * res / crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem],
                                                                         flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem],
                                                            flagval=self.flagvals[elem])
            else:
                del sample[elem]

        return sample

    def __str__(self):
        return 'FixedResize:' + str(self.resolutions)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):
        p = random.random()
        # print(p)

        if p < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class IOGPoints(object):
    """
    Returns the IOG Points (top-left and bottom-right or top-right and bottom-left) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pad_pixel: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """

    def __init__(self, sigma=10, elem='crop_gt', pad_pixel=10, only_bbox_click=False, reverse_channel=False):
        self.sigma = sigma
        self.elem = elem
        self.pad_pixel = pad_pixel
        self.reverse_channel = reverse_channel
        self.only_bbox_click = only_bbox_click

    def __call__(self, sample):

        if sample[self.elem].ndim == 3:
            raise ValueError('IOGPoints not implemented for multiple object per image.')
        _target = sample[self.elem]

        targetshape = _target.shape
        if np.max(_target) == 0:
            # CHANGE
            if self.only_bbox_click:
                sample['IOG_points'] = np.zeros([targetshape[0], targetshape[1], 1],
                                                dtype=_target.dtype)
            else:
                sample['IOG_points'] = np.zeros([targetshape[0], targetshape[1], 2],
                                                dtype=_target.dtype)

        else:
            _points = helpers.iog_points(_target, self.pad_pixel)
            # sample['IOG_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)
            sample['IOG_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, only_bbox_click=self.only_bbox_click,
                                                   reverse_channel=self.reverse_channel)


        return sample

    def __str__(self):
        return 'IOGPoints:(sigma=' + str(self.sigma) + ', pad_pixel=' + str(self.pad_pixel) + ', elem=' + str(
            self.elem) + ')'


class ConcatInputs(object):
# change
    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert (sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]
            res = np.concatenate((res, tmp), axis=2)

        sample['concat'] = res
        return sample

    def __str__(self):
        return 'ClickPoints:' + str(self.elems)


class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """

    def __init__(self, crop_elems=('image', 'gt', 'void_pixels'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(
                            helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(
                            helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop

        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems=' + str(self.crop_elems) + ', mask_elem=' + str(self.mask_elem) + \
               ', relax=' + str(self.relax) + ',zero_pad=' + str(self.zero_pad) + ')'


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """

    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)

        return sample

    def __str__(self):
        return 'ToTensor'
