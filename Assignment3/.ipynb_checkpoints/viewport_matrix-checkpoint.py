import numpy as np

def get_V(right, left, top, bottom):
    x1 = (right - left)/2
    x2 = (right + left)/2
    y1 = (top - bottom)/2
    y2 = (top + bottom)/2

    viewport = np.array([[x1, 0, 0, x2],
                [0, y1, 0, y2],
                [0, 0, 0.5, 0.5],
                [0, 0, 0, 1]])

    return viewport
