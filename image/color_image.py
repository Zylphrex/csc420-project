import cv2 as cv
import numpy as np

import image


class ColorImage(image.Image):
    def __init__(self, img):
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        super().__init__(img)

    def gray(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray = np.round(gray * 255).astype(np.uint8)
        return image.GrayImage(gray)

    def draw_point_groups(self, point_groups):
        img = np.copy(self.img)
        for point_group in point_groups:
            r = np.random.randint(255)
            g = np.random.randint(255)
            b = np.random.randint(255)
            for point in point_group:
                cv.circle(img, (point.x, point.y), 1, (r, g, b))
        return ColorImage(img)

    def draw_lines(self, lines):
        img = np.copy(self.img)
        for line in lines:
            r = np.random.randint(255)
            g = np.random.randint(255)
            b = np.random.randint(255)
            cv.line(img, line.p1.raw, line.p2.raw, (r, g, b), 20)
        return ColorImage(img)

    def draw_quadrilateral(self, quadrilateral):
        img = np.copy(self.img)
        r = np.random.randint(255)
        g = np.random.randint(255)
        b = np.random.randint(255)
        for i in range(4):
            start = quadrilateral.ordered_points[i]
            stop = quadrilateral.ordered_points[(i + 1) % 4]
            cv.line(img, start.raw, stop.raw, (r, g, b), 25)
        return ColorImage(img)
