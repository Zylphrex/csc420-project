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
    if np.abs(group1.direction - group2.direction) > 2 * threshold:
        return False

    if group1.start.x < group2.start.x and group2.stop.x < group1.start.x:
        return False

    if group1.start.y < group2.start.y and group2.stop.y < group1.start.y:
        return False

    if group2.start.x < group1.start.x and group1.stop.x < group2.start.x:
        return False

    if group2.start.y < group1.start.y and group1.stop.y < group2.start.y:
        return False

    distance1 = geometry.manhattan(group1.start, group2.start)
    distance2 = geometry.manhattan(group1.start, group2.stop)
    distance3 = geometry.manhattan(group1.stop, group2.start)
    distance4 = geometry.manhattan(group1.stop, group2.stop)

    good_distance = 2
    very_close = any([
        distance1 < good_distance,
        distance2 < good_distance,
        distance3 < good_distance,
        distance4 < good_distance,
    ])
    if very_close:
        return True

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

    db = abs(b1 - b2)
    if db >= 5:
        return False

    max_x = bounds[0]
    c1 = m1 * max_x + b1
    c2 = m2 * max_x + b2
    dc = abs(c1 - c2)
    if dc >= 5:
        return False

    if db >= 2 and dc >= 2 and abs(db - dc) < 0.5:
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
