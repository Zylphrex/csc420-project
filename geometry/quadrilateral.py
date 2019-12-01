from functools import lru_cache
from itertools import combinations

import numpy as np

import geometry


class Quadrilateral(object):
    def __init__(self, lines, points, bounds):
        assert len(lines) == 4
        assert len(points) == 4

        self.lines = lines
        self.points = points
        self.bounds = bounds

    def score(self):
        dx = self.points[0].x - self.points[2].x
        dy = self.points[0].y - self.points[2].y
        d1 = np.sqrt(dx * dx + dy * dy)

        dx = self.points[1].x - self.points[3].x
        dy = self.points[1].y - self.points[3].y
        d2 = np.sqrt(dx * dx + dy * dy)

        if min(d1, d2) / max(d1, 2) < 0.2:
            return 0

        score = np.sqrt(self.area() / (self.bounds[0] * self.bounds[1]))
        if score <= 0.2:
            return 0

        endpoints = [(0, 1), (2, 3), (0, 2), (1, 3)]
        for line, endpoint in zip(self.lines, endpoints):
            group = geometry.PointGroup()
            group.add(self.points[endpoint[0]])
            group.add(self.points[endpoint[1]])
            side = geometry.Line.from_points(group)

            length = len(side)

            if length <= 25:
                return 0

            coverage = length
            overage = 0

            if side.p1.x < line.p1.x:
                min_point = side.p1
                coverage -= geometry.euclidean(side.p1, line.p1)
            else:
                min_point = line.p1
                overage += geometry.euclidean(side.p1, line.p1)

            if side.p2.x > line.p2.x:
                max_point = side.p2
                coverage -= geometry.euclidean(side.p2, line.p2)
            else:
                max_point = line.p2
                overage += geometry.euclidean(side.p2, line.p2)
            over_length = geometry.euclidean(min_point, max_point)

            multiplier = 1
            if over_length > 0:
                if overage / length > 0.25:
                    return 0
                multiplier -= 0.2 * overage / over_length

            if length > 0:
                coverage_score = max(0, coverage) / length
                if coverage_score < 0.2:
                    return 0
                additional_score = multiplier * coverage_score
                n = 2
                scaled_score = ((n * additional_score) ** 2) / (n * n)
                score += scaled_score

        return score

    def area(self):
        dx = self.points[1].x - self.points[3].x
        dy = self.points[1].y - self.points[3].y
        d = np.sqrt(dx * dx + dy * dy)

        xs = [self.points[1].x, self.points[3].x]
        ys = [self.points[1].y, self.points[3].y]
        a, c = geometry.Line.compute_model(xs, ys)
        b = -1

        denom = np.sqrt(a * a + b * b)
        d1 = np.abs(a * self.points[0].x + b * self.points[0].y + c) / denom
        d2 = np.abs(a * self.points[2].x + b * self.points[2].y + c) / denom
        return d * (d1 + d2) / 2

    @property
    @lru_cache(maxsize=1)
    def ordered_points(self):
        ordered_points = []

        top1, top1_y = None, float('inf')
        for point in self.points:
            if point.y < top1_y:
                top1_y = point.y
                top1 = point

        top2, top2_y = None, float('inf')
        for point in self.points:
            if point is top1:
                continue
            if point.y < top2_y:
                top2_y = point.y
                top2 = point

        bottom1, bottom1_y = None, 0
        for point in self.points:
            if point is top1 or point is top2:
                continue
            if point.y > bottom1_y:
                bottom1_y = point.y
                bottom1 = point

        bottom2, bottom2_y = None, 0
        for point in self.points:
            if point is top1 or point is top2 or point is bottom1:
                continue
            if point.y > bottom2_y:
                bottom2_y = point.y
                bottom2 = point

        if top1.x < top2.x:
            ordered_points.append(top1)
            ordered_points.append(top2)
        else:
            ordered_points.append(top2)
            ordered_points.append(top1)

        if bottom1.x < bottom2.x:
            ordered_points.append(bottom2)
            ordered_points.append(bottom1)
        else:
            ordered_points.append(bottom1)
            ordered_points.append(bottom2)

        return ordered_points

    @property
    def raw(self):
        return [point.raw for point in self.ordered_points]

    def __repr__(self):
        return repr(self.points)

    @staticmethod
    def from_lines(lines, bounds):
        pair1, pair2 = _parallelish_pairs(lines)
        intersections = _compute_intersections(pair1, pair2, bounds)
        return Quadrilateral(pair1 + pair2, intersections, bounds)


def _parallelish_pairs(lines):
    assert len(lines) == 4

    pair1 = None
    closest_angle = float('inf')
    for line1, line2 in combinations(lines, 2):
        angle = geometry.angle_between(line1, line2)
        if angle < closest_angle:
            pair1 = {line1, line2}
            closest_angle = angle

    pair2 = set(lines) - pair1

    return list(pair1), list(pair2)


class OutOfImageBoundaryException(Exception):
    pass


class InsufficientIntersectionsException(Exception):
    pass


def _compute_intersections(pair1, pair2, bounds):
    max_x, max_y = bounds

    points = []
    for line1 in pair1:
        for line2 in pair2:
            try:
                x = (line1.b - line2.b) / (line2.m - line1.m)
                y = line1.m * x + line1.b
                x, y = int(round(x)), int(round(y))
            except (OverflowError, FloatingPointError):
                raise OutOfImageBoundaryException()

            if x < 0 or x > max_x or y < 0 or y > max_y:
                raise OutOfImageBoundaryException()

            points.append(geometry.Point(x, y))

    if len(points) < 4:
        raise InsufficientIntersectionsException()

    return points
