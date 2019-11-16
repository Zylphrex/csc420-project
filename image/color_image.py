import cv2 as cv
import numpy as np

import image


class ColorImage(image.Image):
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
                img[point.y, point.x, 0] = r
                img[point.y, point.x, 1] = g
                img[point.y, point.x, 2] = b
        return ColorImage(img)

    def draw_lines(self, lines):
        img = np.copy(self.img)
        for line in lines:
            r = np.random.randint(255)
            g = np.random.randint(255)
            b = np.random.randint(255)
            cv.line(img, line.p1.raw, line.p2.raw, (r, g, b), 5)
        return ColorImage(img)

    def draw_quadrilateral(self, quadrilateral):
        img = np.copy(self.img)
        r = np.random.randint(255)
        g = np.random.randint(255)
        b = np.random.randint(255)
        for i in range(4):
            start = quadrilateral.ordered_points[i]
            stop = quadrilateral.ordered_points[(i + 1) % 4]
            cv.line(img, start.raw, stop.raw, (r, g, b), 5)
        return ColorImage(img)
