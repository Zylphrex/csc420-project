import numpy as np

import geometry


class BoundedGradientPointGroup(geometry.GradientPointGroup):
    def __init__(self, threshold=geometry.deg_to_rad(15)):
        super().__init__()

        self.threshold = threshold

        self.min_direction = None
        self.max_direction = None

    def compute_equivalent_angle(self, radian):
        radians = np.arange(-2, 3) * np.pi + radian
        deltas = np.abs(radians - self.direction)
        return radians[np.argmin(deltas)]

    def hypothesize_model(self, radian):
        new_min = min(radian, self.min_direction)
        new_max = max(radian, self.max_direction)
        return new_min, new_max

    def fits(self, gradient_direction):
        if self.direction is None:
            return True

        radian = self.compute_equivalent_angle(gradient_direction)
        new_dir = self.compute_new_direction(radian)
        new_min, new_max = self.hypothesize_model(radian)
        return abs(new_max - new_dir) <= self.threshold and \
               abs(new_min - new_dir) <= self.threshold


    def add(self, point, gradient_direction):
        super().add(point, gradient_direction)

        if self.min_direction is None or self.max_direction is None:
            self.min_direction = gradient_direction
            self.max_direction = gradient_direction

        else:
            radian = self.compute_equivalent_angle(gradient_direction)
            new_min, new_max = self.hypothesize_model(radian)
            self.min_direction = new_min
            self.max_direction = new_max

    def __mul__(self, n):
        group = BoundedGradientPointGroup(threshold=self.threshold)
        for p in self.points:
            group.add(p * 6, self.direction)
        group.min_direction = self.min_direction
        group.max_direction = self.max_direction
        return group
