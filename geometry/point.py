import math


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def raw(self):
        return (self.x, self.y)

    def copy(self):
        return Point(self.x, self.y)

    def __mul__(self, n):
        return Point(self.x * n, self.y * n)

    def __rmul__(self, n):
        return self.__mul__(n)

    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)


def euclidean(point1, point2):
    dx = point1.x - point2.x
    dy = point1.y - point2.y
    return math.sqrt(dx * dx + dy * dy)


def manhattan(point1, point2):
    dx = point1.x - point2.x
    dy = point1.y - point2.y
    return abs(dx) + abs(dy)
