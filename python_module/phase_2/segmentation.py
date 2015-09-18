__author__ = 'pablo'
import numpy as np


def img_binarization(img, thr):
    """
    Binarize an image given a certain threshold.

    :param img: Image to binarize
    :type img: ndarray

    :param thr: Threshold of intensity to binarize the image.
    :type thr: float from 0 to 255

    :return: binarized image
    """
    m, n = img.shape

    bin_img = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            bin_img[i, j] = 0 if img[i, j] > thr else 1

    return bin_img


def segmentation(img, bin_thr=240):
    """

    :param img:
    :param bin_thr:
    :return:
    """
    bin_img = img_binarization(img, bin_thr)

    x_sum = bin_img.sum(axis=0)
    y_sum = bin_img.sum(axis=1)

    x_pos = np.where(x_sum != 0)
    x_min = x_pos[0][0]
    x_max = x_pos[0][-1]

    y_pos = np.where(y_sum != 0)
    y_min = y_pos[0][0]
    y_max = y_pos[0][-1]

    m, n = img.shape
    mask = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            if x_min <= j <= x_max and y_min <= i <= y_max:
                mask[i, j] = 1
            else:
                mask[i, j] = 0

    seg_img = img[y_min:y_max+1, x_min:x_max+1]

    return mask, seg_img
