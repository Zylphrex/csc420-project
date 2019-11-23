import cv2 as cv
import numpy as np
from scipy.linalg import solve

import algorithm
import geometry
import image

def find_homography(src_points, dst_points):
    x1 = src_points[0][0]
    y1 = src_points[0][1]

    x2 = src_points[1][0]
    y2 = src_points[1][1]

    x3 = src_points[2][0]
    y3 = src_points[2][1]

    x4 = src_points[3][0]
    y4 = src_points[3][1]

    x1_prime = dst_points[0][0]
    y1_prime = dst_points[0][1]

    x2_prime = dst_points[1][0]
    y2_prime = dst_points[1][1]

    x3_prime = dst_points[2][0]
    y3_prime = dst_points[2][1]

    x4_prime = dst_points[3][0]
    y4_prime = dst_points[3][1]

    p = []
    p.append([-1 * x1, -1 * y1, -1, 0, 0, 0, x1 * x1_prime, y1 * x1_prime, x1_prime])
    p.append([0, 0, 0, -1 * x1, -1 * y1, -1, x1 * y1_prime, y1 * y1_prime, y1_prime])

    p.append([-1 * x2, -1 * y2, -1, 0, 0, 0, x2 * x2_prime, y2 * x2_prime, x2_prime])
    p.append([0, 0, 0, -1 * x2, -1 * y2, -1, x2 * y2_prime, y2 * y2_prime, y2_prime])

    p.append([-1 * x3, -1 * y3, -1, 0, 0, 0, x3 * x3_prime, y3 * x3_prime, x3_prime])
    p.append([0, 0, 0, -1 * x3, -1 * y3, -1, x3 * y3_prime, y3 * y3_prime, y3_prime])

    p.append([-1 * x4, -1 * y4, -1, 0, 0, 0, x4 * x4_prime, y4 * x4_prime, x4_prime])
    p.append([0, 0, 0, -1 * x4, -1 * y4, -1, x4 * y4_prime, y4 * y4_prime, y4_prime])

    p.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

    b = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    x = solve(p, b)

    h = []
    h.append([x[0], x[1], x[2]])
    h.append([x[3], x[4], x[5]])
    h.append([x[6], x[7], x[8]])

    return h
