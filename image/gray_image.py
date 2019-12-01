import numpy as np

import image
import image.gradient as gradient


class GrayImage(image.Image):
    def gradient(self):
        return gradient.ImageGradient(self.img)

    def threshold(self, val):
        img = self.img
        thresh = img >= val
        return GrayImage((thresh * 255).astype(np.uint8))
