__author__ = 'pablo'
import numpy as np


def split_image(img, shape):
    m, n = shape

    row_split = img.shape[0] / m
    col_split = img.shape[1] / n

    splited_images = []

    for i in range(m):
        for j in range(n):
            splited_images.append(img[i*row_split:(i+1)*row_split, j*col_split:(j+1)*col_split])

    return splited_images