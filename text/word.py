import cv2 as cv

import image
import geometry


class Word(object):
    def __init__(self, img, threshold=5):
        self.img = img
        self.threshold = threshold
        self.top_left = None
        self.bottom_right = None

    def fits(self, point):
        if self.top_left is None or self.bottom_right is None:
            return True

        if geometry.euclidean(point, self.top_left) < self.threshold:
            return True

        if geometry.euclidean(point, self.bottom_right) < self.threshold:
            return True

        top_right = geometry.Point(self.bottom_right.x, self.top_left.y)
        if geometry.euclidean(point, top_right) < self.threshold:
            return True

        bottom_left = geometry.Point(self.top_left.x, self.bottom_right.y)
        if geometry.euclidean(point, bottom_left) < self.threshold:
            return True

        return False


    def add(self, point):
        if self.top_left is None or self.bottom_right is None:
            self.top_left = point.copy()
            self.bottom_right = point.copy()
        else:
            self.top_left.x = min(self.top_left.x, point.x)
            self.top_left.y = min(self.top_left.y, point.y)
            self.bottom_right.x = max(self.bottom_right.x, point.x)
            self.bottom_right.y = max(self.bottom_right.y, point.y)


    def crop(self, img=None, pad=5):
        if img is None:
            img = self.img

        if len(img.shape) == 2:
            cropped = img[self.min_y-pad:self.max_y+pad, self.min_x-pad:self.max_x+pad]
        else:
            cropped = img[self.min_y-pad:self.max_y+pad, self.min_x-pad:self.max_x+pad, :]

        return image.ColorImage(cropped)


    def visualize_bounds(self, img=None):
        min_x = self.top_left.x
        min_y = self.top_left.y
        max_x = self.bottom_right.x
        max_y = self.bottom_right.y

        if img is None:
            img = self.img.copy()

        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        cv.line(img, (min_x, min_y), (min_x, max_y), (0, 255, 0), 1)
        cv.line(img, (min_x, min_y), (max_x, min_y), (0, 255, 0), 1)
        cv.line(img, (max_x, max_y), (min_x, max_y), (0, 255, 0), 1)
        cv.line(img, (max_x, max_y), (max_x, min_y), (0, 255, 0), 1)

        return image.ColorImage(img)

    @property
    def min_x(self):
        return self.top_left.x

    @property
    def min_y(self):
        return self.top_left.y

    @property
    def max_x(self):
        return self.bottom_right.x

    @property
    def max_y(self):
        return self.bottom_right.y
