import numpy as np

from viewport_matrix import get_V
from perspective_projection_matrix import get_perspective, get_P
from morphable_model import sample_face, read_pca_model, U
from rotation_matrix import get_rotation_matrix


def pinhole_camera_model(alpha, delta, w, t):
    pca = read_pca_model()
    G = sample_face(pca, alpha, delta)

    matrix = np.array([[G[0]],
                        [G[1]],
                        [G[2]],
                        [1]])

    W = 255
    H = 255

    image_aspect_ratio = W / H
    angle = 10
    near = 300
    far = 2000

    right, left, top, bottom = get_perspective(image_aspect_ratio, angle, near, far)

    V = get_V(right, left, top, bottom)
    P = get_P(near, far, right, left, top, bottom)

    PI = np.dot(V, P)

    R = get_rotation_matrix(w)

    T = np.array([[R[0][0], R[0][1], R[0][2], t[0]],
        [R[1][0], R[1][1], R[1][2], t[1]],
        [R[2][0], R[2][1], R[2][2], t[2]],
        [0, 0, 0, 1]])

    result = np.dot(PI, np.dot(T, matrix))

    print(result)

alpha = U(i=1)
delta = U(i=1)
w = [10, 10, 10]
t = [0, 0, 0]
pinhole_camera_model(alpha, delta, w, t)
