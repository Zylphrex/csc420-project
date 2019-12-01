from itertools import combinations

import numpy as np

import geometry
import text


def detect_point_groups(edge_img, gradient_direction):
    point_groups = []
    visited = np.zeros_like(edge_img).astype(np.bool)
    neighbours = {
        ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1),
        (-1,  0), (-1, -1), ( 0, -1), ( 1, -1),
    }

    for y, x in zip(*np.where(edge_img)):
        if visited[y, x]:
            continue

        point_group = geometry.BoundedGradientPointGroup()
        frontier = [geometry.Point(x, y)]

        while frontier:
            point = frontier.pop()
            direction = gradient_direction[point.y, point.x]

            if not point_group.fits(direction):
                continue

            visited[point.y, point.x] = True
            point_group.add(point, direction)

            for dx, dy in neighbours:
                new_x = point.x + dx
                new_y = point.y + dy
                try:
                    if not edge_img[new_y, new_x]:
                        continue

                    if visited[new_y, new_x]:
                        continue

                    frontier.append(geometry.Point(new_x, new_y))
                except IndexError:
                    continue

        point_groups.append(point_group)

    return point_groups


def detect_quadrilateral(img, lines):
    best_fit = None
    best_score = -float('inf')

    line_groups = combinations(lines, 4)
    bounds = list(reversed(img.img.shape[:2]))
    for lines in line_groups:
        try:
            quadrilateral = geometry.Quadrilateral.from_lines(lines, bounds)
        except geometry.OutOfImageBoundaryException:
            continue
        except geometry.InsufficientIntersectionsException:
            continue

        score = quadrilateral.score()
        if score <= best_score:
            continue

        best_score = score
        best_fit = quadrilateral

    return best_fit


def detect_words(pixels):
    visited = np.zeros_like(pixels).astype(np.bool)
    neighbours = {
        ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1),
        (-1,  0), (-1, -1), ( 0, -1), ( 1, -1),
    }

    words = []

    for y, x in zip(*np.where(pixels)):
        if visited[y, x]:
            continue

        word = text.Word(pixels)
        frontier = [geometry.Point(x, y)]

        while frontier:
            point = frontier.pop()
            if not word.fits(point):
                continue

            visited[point.y, point.x] = True
            word.add(point)

            for dx, dy in neighbours:
                new_x = point.x + dx
                new_y = point.y + dy
                try:
                    if not pixels[new_y, new_x]:
                        continue

                    if visited[new_y, new_x]:
                        continue

                    frontier.append(geometry.Point(new_x, new_y))
                except IndexError:
                    continue

        words.append(word)

    return words
