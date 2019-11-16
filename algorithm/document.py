import cv2 as cv
import numpy as np

import algorithm
import geometry
import image


def detect_document_region(img):
    small_img = img.resize(640, 360)
    gray_img = small_img.gray()

    gradient = gray_img.gradient()
    canny = gradient.canny(100, 200)
    point_groups = algorithm.detect_point_groups(canny, gradient.direction)
    # gradient.imsave('results/canny.png')

    point_groups = list(filter(algorithm.at_least(10), point_groups))
    # small_img.draw_point_groups(point_groups).imsave('results/segments.png')

    point_groups = algorithm.try_merge_point_groups(small_img, point_groups)
    # small_img.draw_point_groups(point_groups).imsave('results/merged.png')

    point_groups = list(filter(algorithm.at_least(50), point_groups))
    to_line = lambda group: geometry.Line.from_points(group * 6)
    lines = list(map(to_line, point_groups))
    # img.draw_lines(lines).imsave('results/lines.png')

    quadrilateral = algorithm.detect_quadrilateral(img, lines)

    return quadrilateral


def warp_document(img, document_region):
    h, w = img.img.shape[:2]
    if h > w:
        target_h, target_w = 1920, 1080
    else:
        target_h, target_w = 1080, 1920

    src_points = np.array([list(point) for point in document_region.raw])
    dst_points = np.array([
        [0, 0], [target_w, 0], [target_w, target_h], [0, target_h]
    ])

    h, _ = cv.findHomography(src_points, dst_points)
    document = cv.warpPerspective(img.img, h, (target_w, target_h))

    return image.ColorImage(document)
