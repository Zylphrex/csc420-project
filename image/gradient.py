from functools import lru_cache

import cv2 as cv
import numpy as np

import image


class ImageGradient(image.Image):
    def __init__(self, img):
        self.source_image = img

    @property
    @lru_cache(maxsize=1)
    def x(self):
        return self._sobel(1, 0)

    @property
    @lru_cache(maxsize=1)
    def y(self):
        return self._sobel(0, 1)

    @property
    @lru_cache(maxsize=1)
    def direction(self):
        opp_adj = np.divide(self.y, self.x, where=self.x!=0)
        opp_adj = np.nan_to_num(np.abs(opp_adj))
        direction = np.arctan(opp_adj)

        idx = np.logical_and(self.x < 0, self.y >= 0)
        direction[idx] = np.pi - direction[idx]

        idx = np.logical_and(self.x < 0, self.y < 0)
        direction[idx] = np.pi + direction[idx]

        idx = np.logical_and(self.x >= 0, self.y < 0)
        direction[idx] = 2 * np.pi - direction[idx]

        return direction

    @lru_cache(maxsize=1)
    def canny(self, low, high):
        self.img = cv.Canny(self.x, self.y, low, high)
        return self.img > 0

    def _sobel(self, x, y, ksize=5, scale=0.1):
        assert x == 1 or x == 0
        assert y == 1 or y == 0

        img = self.source_image
        ddepth = cv.CV_16SC1
        return cv.Sobel(img, ddepth, x, y, ksize=ksize, scale=scale)
