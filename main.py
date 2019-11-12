import collections
import functools
import itertools
import cv2 as cv
import numpy as np
from skimage import color
from skimage import feature
from skimage import io
from skimage import transform
from matplotlib import pyplot as plt


def imread(fname, gray=False):
    # assumes the image resolution is at least 4K
    # then cropped to 4K
    img = io.imread(fname)

    h, w = img.shape[:2]
    if h > w:
        th, tw = 3840, 2160
    else:
        th, tw = 2160, 3840

    # crop to 4K resolution
    dh, dw = (h - th) // 2, (w - tw) // 2
    img = img[dh:th+dh, dw:tw+dw]

    if gray:
        img = color.rgb2gray(img)

    img = (255 * img).astype(np.uint8)
    return img

def resize(img, dim1, dim2):
    dim1, dim2 = max(dim1, dim2), min(dim1, dim2)

    h, w = img.shape[:2]
    if h > w:
        fh, fw = dim1, dim2
    else:
        fh, fw = dim2, dim1

    return cv.resize(img, (fw, fh))


def gradient_direction(img, ksize=(5, 5), scale=1):
    sobel_x = cv.Sobel(img, cv.CV_16SC1, 1, 0 , ksize=5, scale=scale)
    sobel_y = cv.Sobel(img, cv.CV_16SC1, 0, 1 , ksize=5, scale=scale)

    # this calculates the angle between the vector and the (closest) x-axis
    direction = np.arctan(np.abs(np.divide(sobel_y, sobel_x, where=sobel_x!=0)))

    # Q1: nothing to be done here as it is already correct

    # Q2: need to subtract from 180 degrees (pi radians)
    idx = np.logical_and(sobel_x <  0, sobel_y >= 0)
    direction[idx] = np.pi - direction[idx]

    # Q3: need to add 180 degrees (pi radians)
    idx = np.logical_and(sobel_x <  0, sobel_y <  0)
    direction[idx] = np.pi + direction[idx]

    # Q4: need to subtract from 360 degrees (2*pi radians)
    idx = np.logical_and(sobel_x >= 0, sobel_y <  0)
    direction[idx] = 2 * np.pi - direction[idx]

    return sobel_x, sobel_y, direction


class GradientPoint(object):
    def __init__(self, x, y, direction):
        # we only store the location and the direction
        # of the gradient as the magnitude of the
        # gradient is not useful in our computations
        self.x = x
        self.y = y

        # the direction is measured in terms of radians
        # counter clock wise starting from the positive
        # x-axis (basically what you remember from trig)
        self.direction = direction

    def __mul__(self, n):
        return GradientPoint(self.x * n, self.y * n,
                             self.direction)

    def __rmul__(self, n):
        return self.__mul__(n)


class Edge(object):
    def __init__(self, threshold=15 * np.pi / 180):
        # this is the threshold to use to see if new
        # data fits the model of this current edge
        #
        # the thresholding works as follows, whenever we
        # want to see if a new point fits in the model of
        # the edge, we compute a new model as follows
        # 1. compute the nearest equivalent angle to the
        #    average direction of the current edge by
        #    adding/subtracting multiples of pi (180 degrees)
        self.threshold = threshold

        # this is the list of points that are part of
        # this particular edge
        self.points = []

        self.top = None
        self.bottom = None
        self.left = None
        self.right = None

        # this is the average direction of the points
        # present in this edge
        self.direction = None

        # this is the min and max bounds for the direction
        # for the points present in this edge
        self.min_direction = None
        self.max_direction = None

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __mul__(self, n):
        edge = Edge(self.threshold)
        for point in self.points:
            edge.add(point * 6)
        return edge

    def __rmul__(self, n):
        return self.__mul__(n)

    def compute_angle(self, radian):
        radians = np.arange(-2, 3) * np.pi + radian
        deltas = np.abs(radians - self.direction)
        return radians[np.argmin(deltas)]

    def hypothesize_model(self, radian):
        new_min = min(radian, self.min_direction)
        new_max = max(radian, self.max_direction)
        n = len(self.points)
        new_dir = (n * self.direction + radian) / (n + 1)
        return new_dir, new_min, new_max

    def fits(self, data, debug=False):
        if isinstance(data, GradientPoint):
            point = data
            if self.direction is None:
                # when there are no points in the model, it always fits
                return True
            else:
                # this makes the assumption that the new point
                # is connected to the existing points 
                radian = self.compute_angle(point.direction)
                new_dir, new_min, new_max = self.hypothesize_model(radian)
                return new_max - new_dir <= self.threshold and \
                       new_dir - new_min <= self.threshold

        elif isinstance(data, Edge):
            edge = data

            if np.abs(self.direction - edge.direction) > 2 * self.threshold:
                if debug:
                    print('edge direction too different')
                return False

            if self.start.x < edge.start.x and edge.stop.x < self.stop.x:
                return False

            if edge.start.x < self.start.x and self.stop.x < edge.stop.x:
                return False

            if self.start.y < edge.start.y and edge.stop.y < self.stop.y:
                return False

            if edge.start.y < self.start.y and self.stop.y < edge.stop.y:
                return False

            def l1_norm(p1, p2):
                return abs(p1.x - p2.x) + abs(p1.y - p2.y)

            d1 = l1_norm(self.start, edge.start)
            d2 = l1_norm(self.start, edge.stop)
            d3 = l1_norm(self.stop, edge.start)
            d4 = l1_norm(self.stop, edge.stop)

            good_d = 2
            if d1 < good_d or d2 < good_d or d3 < good_d or d4 < good_d:
                if debug:
                    print('edges touching')
                return True

            bad_d = 30
            if d1 > bad_d and d2 > bad_d and d3 > bad_d and d4 > bad_d:
                if debug:
                    print('edges too far apart')
                return False

            def approximate_slope(points):
                xs = np.array([p.x for p in points])
                x_bar = xs.mean()
                X = xs - x_bar
                ys = np.array([p.y for p in points])
                y_bar = ys.mean()
                Y = ys - y_bar
                SXY = np.sum(X * Y)
                SXX = np.sum(X * X)
                slope = np.divide(SXY, SXX, where=SXX!=0)
                return slope, x_bar, y_bar

            self_slope, self_x, self_y = approximate_slope(self.points)
            edge_slope, edge_x, edge_y = approximate_slope(edge.points)

            self_intercept = self_y - self_slope * self_x
            edge_intercept = edge_y - edge_slope * edge_x

            if abs(self_intercept - edge_intercept) > 1:
                if debug:
                    print('edge y-intercepts too far')
                return False

            both_slope, both_x, both_y = approximate_slope(self.points + edge.points)

            if self_slope - self.threshold > both_slope or \
                    self_slope + self.threshold < both_slope:
                if debug:
                    print('slope too different from self')
                return False

            if edge_slope - edge.threshold > both_slope or \
                    edge_slope + edge.threshold < both_slope:
                if debug:
                    print('slope too different from other')
                return False

            return True

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

    def add(self, data):
        if isinstance(data, GradientPoint):
            point = data
            self.points.append(point)
            self.update_bounds(point)
            if self.direction is None:
                # when we don't have any data points in
                # the edge yet, there's no checking to
                # be done and we just add it
                self.direction = point.direction
                self.min_direction = point.direction
                self.max_direction = point.direction

            else:
                radian = self.compute_angle(point.direction)
                new_dir, new_min, new_max = self.hypothesize_model(radian)
                self.direction = new_dir
                self.min_direction = new_min
                self.max_direction = new_max

        elif isinstance(data, Edge):
            edge = data
            for point in edge:
                self.add(point)


class Line(object):
    def __init__(self, slope, intercept, start_x, stop_x):
        self.slope = slope
        self.intercept = intercept
        self.start_x = min(start_x, stop_x)
        self.stop_x = max(start_x, stop_x)

    def __len__(self):
        dx = self.stop_x - self.start_x
        dy = self.stop_y - self.start_y
        return int(round(np.sqrt(dx * dx + dy * dy)))

    def compute(self, x):
        return int(self.slope * x + self.intercept)

    @property
    def start_y(self):
        return self.compute(self.start_x)

    @property
    def stop_y(self):
        return self.compute(self.stop_x)

    @staticmethod
    def from_edge(edge):
        xs = np.array([p.x for p in edge.points])
        x_bar = xs.mean()
        X = xs - x_bar
        ys = np.array([p.y for p in edge.points])
        y_bar = ys.mean()
        Y = ys - y_bar
        SXY = np.sum(X * Y)
        SXX = np.sum(X * X)
        slope = np.divide(SXY, SXX, where=SXX!=0)
        intercept = y_bar - slope * x_bar
        return Line(slope, intercept, edge.start.x, edge.stop.x)

    @staticmethod
    def from_points(x1, y1, x2, y2):
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        denom = x2 - x1
        if denom == 0:
            denom = 1e-10
        slope = (y2 - y1) / denom
        intercept = y1 - slope * x1
        return Line(slope, intercept, x1, x2)


def detect_edges(edge_img, grad_dir, min_length=10):
    edges = []
    visited = np.zeros_like(edge_img).astype(np.bool)
    neighbours = {
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1),
    }

    for y, x in zip(*np.where(edge_img)):
        if visited[y, x]:
            # do use the same point in multiple edges
            continue

        edge = Edge()
        frontier = [GradientPoint(x, y, grad_dir[y, x])]

        while frontier:
            point = frontier.pop()

            if not edge.fits(point):
                # if the point does not fit the model
                # of this particular edge, we move on
                continue

            # if it does fit the model, then we mark the
            # point as visited and add it to the edge
            visited[point.y, point.x] = True
            edge.add(point)

            # then we must add any potential neighbouring eges to the
            # frontier so it can be used to extend the edge
            for dx, dy in neighbours:
                new_x = point.x + dx
                new_y = point.y + dy
                try:
                    if not edge_img[new_y, new_x]:
                        # if there is no edge at the new pixel, then
                        # there's no reason to keep checking
                        continue

                    if visited[new_y, new_x]:
                        # if we've visited the new pixel before, then
                        # there's also no reason to keep checking
                        continue

                    new_d = grad_dir[new_y, new_x]
                    new_point = GradientPoint(new_x, new_y, new_d)
                    frontier.append(new_point)
                except IndexError:
                    # we might get a new_x or new_y that is out of
                    # bounds if we were on a point by the borders
                    continue

        if len(edge) > min_length:
            edges.append(edge)

    return edges


def merge_edges(edges):
    edges_set = set(edges)

    while True:
        done = False
        edges_list = [edge for edge in edges if edge in edges_set]
        pairs = itertools.combinations(edges_list, 2)
        for e1, e2 in pairs:
            if not e1.fits(e2):
                continue

            e1.add(e2)
            edges_set.remove(e2)
            break
        else:
            done = True
        if done:
            break

    return [edge for edge in edges if edge in edges_set]


def detect_lines(img, low=100, high=200, min_length=100):
    grad_x, grad_y, grad_dir = gradient_direction(img)
    canny_edges = cv.Canny(grad_x, grad_y, low, high)
    edges = detect_edges(canny_edges > 0, grad_dir)
    edges = merge_edges(edges)
    edges = list(filter(lambda e: len(e) > min_length, edges))
    return edges


def detect_quadrilateral(img, lines):
    best_shape = None
    best_score = -float('inf')

    edge_groups = itertools.combinations(lines, 4)
    for edges in edge_groups:

        pairs = itertools.combinations(edges, 2)
        pair1 = None
        closest_angle = float('inf')
        for e1, e2 in pairs:
            a1 = np.arctan(e1.slope)
            a2 = np.arctan(e2.slope)
            angle = abs(a1 - a2)
            if angle < closest_angle:
                pair1 = [e1, e2]
                closest_angle = angle

        pair2 = []
        for edge in edges:
            if edge not in pair1:
                pair2.append(edge)

        out_of_bounds = False
        points = []
        for edge1 in pair1:
            for edge2 in pair2:
                x = (edge1.intercept - edge2.intercept) / (edge2.slope - edge1.slope)
                y = edge1.slope * x + edge1.intercept
                try:
                    x, y = int(round(x)), int(round(y))
                    out_of_bounds = x < 0 or x > img.shape[1] or y < 0 or y > img.shape[0]
                except OverflowError:
                    out_of_bounds = True

                if out_of_bounds:
                    break
                points.append((x, y))

            if out_of_bounds:
                break

        if out_of_bounds:
            continue

        score = 0

        endpoints = [(0, 1), (2, 3), (0, 2), (1, 3)]
        for edge, endpoint in zip(pair1 + pair2, endpoints):
            p1 = points[endpoint[0]]
            p2 = points[endpoint[1]]
            line = Line.from_points(*p1, *p2)

            length = l2_norm(line.start_x, line.start_y, line.stop_x, line.stop_y)
            coverage = length
            overage = 0
            if line.start_x < edge.start_x:
                min_x, min_y = line.start_x, line.start_y
                coverage -= l2_norm(line.start_x, line.start_y, edge.start_x, edge.start_y)
            else:
                min_x, min_y = edge.start_x, edge.start_y
                overage += l2_norm(line.start_x, line.start_y, edge.start_x, edge.start_y)
            if line.stop_x > edge.stop_x:
                max_x, max_y = line.stop_x, line.stop_y
                coverage -= l2_norm(line.stop_x, line.stop_y, edge.stop_x, edge.stop_y)
            else:
                max_x, max_y = edge.stop_x, edge.stop_y
                overage += l2_norm(line.stop_x, line.stop_y, edge.stop_x, edge.stop_y)
            over_len = l2_norm(min_x, min_y, max_x, max_y) - length
            coverage = max(0, coverage)
            if over_len > 0:
                score += (coverage / length) * (1 - 0.05 * overage / over_len)
            else:
                score += coverage / length
            score += coverage / length

        if score <= best_score:
            continue

        best_score = score
        best_shape = points

    # make sure to permute this in some order so we can construct the quadrilaterl
    return best_shape


def l2_norm(x1, y1, x2, y2):
    dx = x2 - x1
    dx2 = dx * dx
    dy = y2 - y1
    dy2 = dy * dy
    return np.sqrt(dx2 + dy2)


def visualize_edges(img, edges):
    print('edges: {}'.format(len(edges)))
    def _visualize(img, edge, r=0, g=255, b=0):
        for p in edge:
            img[p.y, p.x, 0] = r
            img[p.y, p.x, 1] = g
            img[p.y, p.x, 2] = b
        return img

    v = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2RGB)
    for edge in edges:
        r = np.random.randint(255)
        g = np.random.randint(255)
        b = np.random.randint(255)
        v = _visualize(v, edge, r, g, b)
    return v


def visualize_lines(img, lines):
    print('lines: {}'.format(len(lines)))
    def _visualize(img, line, r=0, g=255, b=0):
        cv.line(img, (line.start_x, line.start_y), (line.stop_x, line.stop_y), (r, g, b), 5)
        return img

    v = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2RGB)
    for line in lines:
        r = np.random.randint(255)
        g = np.random.randint(255)
        b = np.random.randint(255)
        v = _visualize(v, line, r, g, b)
    return v


def visualize_points(img, points):
    print('points: {}'.format(len(points)))
    def _visualize(img, x, y, r=0, g=255, b=0):
        cv.circle(img, (x, y), 10, (r, g, b), 25, -1)
        return img

    v = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2RGB)
    for x, y in points:
        r = np.random.randint(255)
        g = np.random.randint(255)
        b = np.random.randint(255)
        v = _visualize(v, x, y, r, g, b)
    return v

def main():
    img = imread('images/test3.png', gray=True)
    small_img = resize(img, 640, 360) # reduce to 360p
    edges = detect_lines(small_img)
    io.imsave('results/test1.png', visualize_edges(small_img, edges))
    lines = [Line.from_edge(edge * 6) for edge in edges] # increase resolution to 4K
    io.imsave('results/test2.png', visualize_lines(img, lines))
    quadrilateral = detect_quadrilateral(img, lines)
    io.imsave('results/test3.png', visualize_points(img, quadrilateral))



if __name__ == '__main__':
    main()
