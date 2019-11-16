from functools import lru_cache

import cv2 as cv
import numpy as np


class Image(object):
    def __init__(self, img):
        self.img = img

    def resize(self, dim1, dim2):
        dim1, dim2 = max(dim1, dim2), min(dim1, dim2)

        h, w = self.img.shape[:2]
        if h > w:
            resized_h, resized_w = dim1, dim2
        else:
            resized_h, resized_w = dim2, dim1

        resized_img = cv.resize(self.img, (resized_w, resized_h))
        return self.__class__(resized_img)

    def imsave(self, file_name):
        cv.imwrite(file_name, self.img)

    @classmethod
    def imread(cls, file_name):
        img = cv.imread(file_name)

        h, w = img.shape[:2]
        if h > w:
            target_h, target_w = 3840, 2160
        else:
            target_h, target_w = 2160, 3840

        # crop to 4K resolution
        delta_h = (h - target_h) // 2
        delta_w = (w - target_w) // 2
        img = img[delta_h:target_h + delta_h, delta_w:target_w + delta_w]
        return cls(img)
