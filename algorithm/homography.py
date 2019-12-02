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

    p.append([x1, y1, 1, 0, 0, 0, -1 * x1 * x1_prime, -1 * y1 * x1_prime, -1 * x1_prime])
    p.append([0, 0, 0, x1, y1, 1, -1 * x1 * y1_prime, -1 * y1 * y1_prime, -1 * y1_prime])

    p.append([x2, y2, 1, 0, 0, 0, -1 * x2 * x2_prime, -1 * y2 * x2_prime, -1 * x2_prime])
    p.append([0, 0, 0, x2, y2, 1, -1 * x2 * y2_prime, -1 * y2 * y2_prime, -1 * y2_prime])

    p.append([x3, y3, 1, 0, 0, 0, -1 * x3 * x3_prime, -1 * y3 * x3_prime, -1 * x3_prime])
    p.append([0, 0, 0, x3, y3, 1, -1 * x3 * y3_prime, -1 * y3 * y3_prime, -1 * y3_prime])

    p.append([x4, y4, 1, 0, 0, 0, -1 * x4 * x4_prime, -1 * y4 * x4_prime, -1 * x4_prime])
    p.append([0, 0, 0, x4, y4, 1, -1 * x4 * y4_prime, -1 * y4 * y4_prime, -1 * y4_prime])

    p = np.asarray(p)
    U, S, V = np.linalg.svd(p)
    L = V[-1,:] / V[-1,-1]
    h = L.reshape(3, 3)

    return h

def warp_image(img, h, target):
    target_w = target[0]
    target_h = target[1]

    h = cv.invert(h)[1]   

    h_11 = h[0][0]
    h_12 = h[0][1]
    h_13 = h[0][2]

    h_21 = h[1][0]
    h_22 = h[1][1]
    h_23 = h[1][2]

    h_31 = h[2][0]
    h_32 = h[2][1]
    h_33 = h[2][2]
    
    warped_image = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    for y in range(target_h):
        for x in range(target_w):
            m = np.array([x, y, 1])
            q = h.dot(m)
            x_old = int(round(q[0]/q[2]))
            y_old = int(round(q[1]/q[2]))
            warped_image[y][x] = img[y_old][x_old]

    return warped_image
