from itertools import product

import numpy as np

import geometry
import text


def try_merge_point_groups(img, point_groups):
    bounds = list(reversed(img.img.shape[:2]))
    point_groups = set(point_groups)

    while True:
        done = True

        # iterate every pair of possible point groups
        for group1 in list(iter(point_groups)):
            if group1 not in point_groups:
                continue
            for group2 in list(iter(point_groups)):
                if group1 is group2 or group2 not in point_groups:
                    continue

                # measure similarity between two groups of points
                # and merge them together if it passes the test
                if not _point_groups_are_similar(group1, group2, bounds):
                    continue

                point_groups.remove(group1)
                point_groups.remove(group2)

                group = _merge_two_point_groups(group1, group2)
                point_groups.add(group)
                group1 = group

                done = False

        if done:
            break

    return list(point_groups)


def _point_groups_are_similar(group1, group2, bounds, threshold=geometry.deg_to_rad(15)):
    # measure the nearest distance between the two point groups
    distance1 = geometry.manhattan(group1.start, group2.start)
    distance2 = geometry.manhattan(group1.start, group2.stop)
    distance3 = geometry.manhattan(group1.stop, group2.start)
    distance4 = geometry.manhattan(group1.stop, group2.stop)

    # reject them if they are too far apart
    bad_distance = 50
    too_far = all([
        distance1 > bad_distance,
        distance2 > bad_distance,
        distance3 > bad_distance,
        distance4 > bad_distance,
    ])
    if too_far:
        return False

    # fit a line to each point group
    xs1, ys1 = group1.xs, group1.ys
    m1, b1 = geometry.Line.compute_model(xs1, ys1)
    xs2, ys2 = group2.xs, group2.ys
    m2, b2 = geometry.Line.compute_model(xs2, ys2)

    xs = [
        min(group1.start.x, group2.start.x),
        max(group1.stop.x, group2.stop.x),
    ]

    # reject them if the slopes and y-intercepts are too far apart
    for x in xs:
        y1 = m1 * x + b1
        y2 = m2 * x + b2
        dy = abs(y1 - y2)
        if dy >= 5:
            return False

    # fit a line to the merged point group
    m, b = geometry.Line.compute_model(xs1 + xs2, ys1 + ys2)

    # reject if the fitted line is too different from either of the
    # original point groups
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


def try_merge_words(words):
    words = set(words)

    while True:
        done = True

        # iterate every pair of possible words
        for w1 in list(iter(words)):
            if w1 not in words:
                continue

            for w2 in list(iter(words)):
                if w1 is w2 or w2 not in words:
                    continue

                # if the bounding boxes of the words are physically
                # too far apart
                if not _words_are_close(w1, w2):
                    continue

                words.remove(w1)
                words.remove(w2)

                word = _merge_two_words(w1, w2)
                words.add(word)
                w1 = word

                done = False

        if done:
            break

    return [word for word in words if word.area >= 250]


def _overlap(w1, w2):
    if w1.min_x > w2.max_x or w2.min_x > w1.max_x:
        return False

    if w1.min_y > w2.max_y or w2.min_y > w1.max_y:
        return False

    return True


def _words_are_close(w1, w2, threshold_x=7, threshold_y=15):
    # overlapping bounding boxes means we should merge
    if _overlap(w1, w2):
        return True

    w1x = [w1.min_x, w1.max_x]
    w1y = [w1.min_y, w1.max_y]
    w2x = [w2.min_x, w2.max_x]
    w2y = [w2.min_y, w2.max_y]
    dx = float('inf')
    dy = float('inf')

    # if they arent too far offset from each other
    for x1, y1, x2, y2 in product(w1x, w1y, w2x, w2y):
        dx = min(dx, abs(x1 - x2))
        dy = min(dy, abs(y1 - y2))

    return dx < threshold_x and dy < threshold_y


def _merge_two_words(w1, w2):
    word = text.Word(w1.img)
    word.add(w1.top_left)
    word.add(w1.bottom_right)
    word.add(w2.top_left)
    word.add(w2.bottom_right)
    return word
