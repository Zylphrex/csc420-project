import numpy as np

import geometry


class GradientPointGroup(geometry.PointGroup):
    def __init__(self):
        super().__init__()
        self.direction = None

    def compute_equivalent_angle(self, radian):
        radians = np.arange(-2, 3) * np.pi + radian
        deltas = np.abs(radians - self.direction)
        return radians[np.argmin(deltas)]

    def compute_new_direction(self, radian):
        n = len(self.points)
        return (n * self.direction + radian) / (n + 1)

    def add(self, point, gradient_direction):
        super().add(point)

        if self.direction is None:
            self.direction = gradient_direction
        else:
            radian = self.compute_equivalent_angle(gradient_direction)
            self.direction = self.compute_new_direction(radian)

    def __mul__(self, n):
        group = GradientPointGroup()
        for p in self.points:
            group.add(p * 6, self.direction)
        return group
