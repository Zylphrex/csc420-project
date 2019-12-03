import datetime

import cv2 as cv
import numpy as np

import algorithm
import geometry
import image


def detect_document_region(img):
    # scale image down for faster edge detection
    small_img = img.resize(640, 360)
    gray_img = small_img.gray()
    gray_img.imsave('results/small_gray.png')

    # compute the image gradients
    gradient = gray_img.gradient()
    # compute canny edges given the gradient information
    canny = gradient.canny(100, 200)
    # step 1.1 cluster edge points into segments based on direction information
    point_groups = algorithm.detect_point_groups(canny, gradient.direction)
    # visualize canny edges
    gradient.imsave('results/canny.png')

    # eliminate segments that are too short
    point_groups = list(filter(algorithm.at_least(10), point_groups))
    # visualize line segments
    small_img.draw_point_groups(point_groups).imsave('results/segments.png')

    # step 1.2 merge line segments into longer lines
    point_groups = algorithm.try_merge_point_groups(small_img, point_groups)
    # visualize longer lines
    small_img.draw_point_groups(point_groups).imsave('results/merged.png')

    # eliminate lines that are too short
    point_groups = list(filter(algorithm.at_least(50), point_groups))

    # scale image backup to 4K resolution
    to_line = lambda group: geometry.Line.from_points(group * 6)
    # fit an equation of a line to each group of points
    lines = list(map(to_line, point_groups))
    # visualize the lines
    img.draw_lines(lines).imsave('results/lines.png')

    # step 1.3 search for the best scoring quadrilateral in the space
    # of possible quadrilaterals
    quadrilateral = algorithm.detect_quadrilateral(img, lines)

    return quadrilateral


def warp_document(img, document_region, region_w=1080, region_l=1920, pad=10):
    h, w = img.img.shape[:2]
    if h > w:
        target_h, target_w = region_l, region_w
    else:
        target_h, target_w = region_w, region_l

    target_h += 2 * pad
    target_w += 2 * pad

    src_points = np.array([list(point) for point in document_region.raw])
    dst_points = np.array([
        [1, 1], [target_w, 1], [target_w, target_h], [1, target_h]
    ])

    # project the document onto a flat image plane through homography
    h = algorithm.find_homography(src_points, dst_points)
    document = algorithm.warp_image(img.img, h, (target_w, target_h))

    return image.ColorImage(document[pad:-pad, pad:-pad, :])
