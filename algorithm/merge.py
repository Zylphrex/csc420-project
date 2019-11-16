from itertools import combinations

import numpy as np

import geometry


def try_merge_point_groups(img, point_groups):
    bounds = list(reversed(img.img.shape[:2]))
    point_groups = set(point_groups)

    while True:
        done = False

        pairs = combinations(point_groups, 2)

        for group1, group2 in pairs:
            if not _point_groups_are_similar(group1, group2, bounds):
                continue

            point_groups.remove(group1)
            point_groups.remove(group2)

            group = _merge_two_point_groups(group1, group2)
            point_groups.add(group)
            break
        else:
            done = True

        if done:
            break

    return list(point_groups)


def _point_groups_are_similar(group1, group2, bounds, threshold=geometry.deg_to_rad(15)):
    distance1 = geometry.manhattan(group1.start, group2.start)
    distance2 = geometry.manhattan(group1.start, group2.stop)
    distance3 = geometry.manhattan(group1.stop, group2.start)
    distance4 = geometry.manhattan(group1.stop, group2.stop)


    bad_distance = 50
    too_far = all([
        distance1 > bad_distance,
        distance2 > bad_distance,
        distance3 > bad_distance,
        distance4 > bad_distance,
    ])
    if too_far:
        return False


    xs1, ys1 = group1.xs, group1.ys
    m1, b1 = geometry.Line.compute_model(xs1, ys1)
    xs2, ys2 = group2.xs, group2.ys
    m2, b2 = geometry.Line.compute_model(xs2, ys2)

    xs = [
        min(group1.start.x, group2.start.x),
        max(group1.stop.x, group2.stop.x),
    ]

    for x in xs:
        y1 = m1 * x + b1
        y2 = m2 * x + b2
        dy = abs(y1 - y2)
        if dy >= 5:
            return False

    m, b = geometry.Line.compute_model(xs1 + xs2, ys1 + ys2)

    if m1 - threshold > m or m1 + threshold < m:
        return False

    if m2 - threshold > m or m2 + threshold < m:
        return False

    return True


def _merge_two_point_groups(group1, group2):
    group = geometry.GradientPointGroup()
    for point in group1:
        group.add(point, group1.direction)
    for point in group2:
        group.add(point, group2.direction)
    return group
