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

    h_inv = np.linalg.inv(h)
    coords = np.empty((3, target_h, target_w))
    for x in range(target_w):
        for y in range(target_h):
            coords[:, y, x] = x, y, 1

    coords = h_inv.dot(coords.reshape((3, -1)))
    coords = coords[:2, :] / coords[2, :]
    coords = np.round(coords).astype(np.int)
    return img[coords[1], coords[0]].reshape((target_h, target_w, 3))
