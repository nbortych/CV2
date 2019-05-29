import numpy as np
import math

def get_perspective(image_aspect_ratio, angle, near, far):
    scale = math.tan((angle * 0.5 * math.pi) / 180) * near
    right = image_aspect_ratio * scale
    left = -right
    top = scale
    bottom = -top

    return right, left, top, bottom


def get_P(near, far, right, left, top, bottom):
    x1 = (2 * near)/(right - 1)
    x2 = (right + left)/(right - left)
    x3 = (2 * near)/(top - bottom)
    x4 = (top + bottom)/(top - bottom)
    x5 = -(far + near)/(far - near)
    x6 = -(2 * far * near)/(far - near)
    x7 = -1

    P = np.array([[x1, 0, x2, 0],
        [0, x3, x4, 0],
        [0, 0, x5, x6],
        [0, 0, x7, 0]])

    return P
