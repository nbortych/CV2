import numpy as np

from viewport_matrix import get_V
from perspective_projection_matrix import get_perspective, get_P
from morphable_model import sample_face, read_pca_model, U
from rotation_matrix import get_rotation_matrix

from data_def import Mesh
from mesh_to_png import triangles, mean_tex, mesh_to_png

def rotate_face(angles):
    """
    :param angles: list of angles. each angle has three entries [theta_x, theta_y, theta_z]
    :return:
    """
    # sample face
    pca = read_pca_model()
    G = sample_face(pca).T

    # transform to homogeneous coordinates
    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

    for w in angles:
        # get rotation matrix
        T = np.eye(4)
        T[:3, :3] = get_rotation_matrix(w)

        # save resulting rotated face
        mesh = Mesh(vertices=(T @ G_h)[:3].T, colors=mean_tex, triangles=triangles)
        mesh_to_png("./results/rotation/"+str(w)+".png", mesh)

    return

def pinhole_camera_model(w, t):

    # sample face
    pca = read_pca_model()
    G = sample_face(pca).T

    # transform to homogeneous coordinates
    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

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

    return


# Task 3.1
# Result are saved in results/rotation
w = [[0,10,0], [0,0,0], [0,-10,0]]
rotate_face(w)

