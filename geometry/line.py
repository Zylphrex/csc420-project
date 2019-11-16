import numpy as np

import geometry


class Line(object):
    def __init__(self, m, b, x1, x2):
        self.m = m
        self.b = b

        x1, x2 = min(x1, x2), max(x1, x2)

        y1 = int(round(m * x1 + b))
        x1 = int(round(x1))
        self.p1 = geometry.Point(x1, y1)

        y2 = int(round(m * x2 + b))
        x2 = int(round(x2))
        self.p2 = geometry.Point(x2, y2)

    def __len__(self):
        return int(round(geometry.euclidean(self.p1, self.p2)))

    def __repr__(self):
        sign = '+' if self.b >= 0 else '-'
        formula = 'y = {} x {} {}'.format(self.m, sign, abs(self.b))
        return '{} ({}, {})'.format(formula, self.p1, self.p2)

    @staticmethod
    def compute_model(xs, ys):
        xs = np.array(xs)
        x_bar = xs.mean()
        X = xs - x_bar
        ys = np.array(ys)
        y_bar = ys.mean()
        Y = ys - y_bar
        SXY = np.sum(X * Y)
        SXX = np.sum(X * X)
        m = np.divide(SXY, SXX, where=SXX!=0)
        b = y_bar - m * x_bar
        return m, b

    @staticmethod
    def from_points(points):
        m, b = Line.compute_model(points.xs, points.ys)
        return Line(m, b, points.start.x, points.stop.x)


def angle_between(line1, line2):
    angle1 = np.arctan(line1.m)
    angle2 = np.arctan(line2.m)
    return abs(angle1 - angle2)
