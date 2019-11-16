import numpy as np

import geometry


class PointGroup(object):
    def __init__(self, threshold=geometry.deg_to_rad(15)):
        self.points = []
        self.cache = {}

        self.top = None
        self.bottom = None
        self.left = None
        self.right = None

    def add(self, point):
        self.points.append(point)
        self.cache = {}
        self.update_bounds(point)

    @property
    def xs(self):
        if 'xs' not in self.cache:
            self.cache['xs'] = [p.x for p in self.points]
        return self.cache['xs']

    @property
    def ys(self):
        if 'ys' not in self.cache:
            self.cache['ys'] = [p.y for p in self.points]
        return self.cache['ys']

    def update_bounds(self, point):
        if self.top is None or point.y < self.top.y:
            self.top = point

        if self.bottom is None or point.y > self.bottom.y:
            self.bottom = point

        if self.left is None or point.x < self.left.x:
            self.left = point

        if self.right is None or point.x > self.right.x:
            self.right = point

    @property
    def start(self):
        dx = self.right.x - self.left.x
        dy = self.bottom.y - self.top.y
        if dx > dy:
            return self.left
        else:
            return self.top

    @property
    def stop(self):
        dx = self.right.x - self.left.x
        dy = self.bottom.y - self.top.y
        if dx > dy:
            return self.right
        else:
            return self.bottom

    def __mul__(self, n):
        group = PointGroup(threshold=self.threshold)
        for p in self.points:
            group.add(p * 6)
        return group

    def __rmul__(self, n):
        return self.__mul__(n)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return repr(self.points)
