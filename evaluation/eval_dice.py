import os.path
import cv2
import numpy as np
from PIL import Image

import dataloaders.helpers as helpers
import evaluation.evaluation as evaluation

from medpy import metric

def calculate_metric_percase(pred, gt):
        # dice = metric.binary.dc(pred, gt)
        # jc = metric.binary.jc(pred, gt)
        # hd = metric.binary.hd95(pred, gt)
    # if 0 == np.count_nonzero(pred):
    #     return 0

    asd = metric.binary.asd(pred, gt)
        # return dice, jc, hd, asd
    return asd


def eval_one_result(loader, folder, one_mask_per_image=False, mask_thres=0.5, use_void_pixels=True, custom_box=False):

    # Allocate
    eval_result = dict()

    eval_result["all_dice"] = np.zeros(len(loader))
    eval_result["all_asd"] = np.zeros(len(loader))

    # Iterate
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        if not one_mask_per_image:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
        else:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '.png')

        mask = np.array(Image.open(filename)).astype(np.float32) / 255.
        gt = np.squeeze(helpers.tens2image(sample["gt"]))
        if use_void_pixels:
            void_pixels = np.squeeze(helpers.tens2image(sample["void_pixels"]))
        if mask.shape != gt.shape:
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)

        # Threshold
        mask = (mask > mask_thres)
        if use_void_pixels:
            void_pixels = (void_pixels > 0.5)

        eval_result["all_dice"][i] = evaluation.dice(gt, mask)
        # eval_result["all_asd"][i] = calculate_metric_percase(mask, gt)

    return eval_result





