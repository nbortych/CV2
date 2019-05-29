import numpy as np

from viewport_matrix import get_V
from perspective_projection_matrix import get_perspective, get_P
from morphable_model import get_face_point_cloud, read_pca_model, U, random_face_point_cloud
from rotation_matrix import get_rotation_matrix

from data_def import Mesh
from mesh_to_png import triangles, mean_tex, mesh_to_png
import matplotlib.pyplot as plt

def rotate_face(angles):
    """
    :param angles: list of angles. each angle has three entries [theta_x, theta_y, theta_z]
    :return:
    """
    # sample face
    pca = read_pca_model()
    G = random_face_point_cloud(pca).T

    # transform to homogeneous coordinates
    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

    for w in angles:
        # get T matrix for only rotation
        T = np.eye(4)
        T[:3, :3] = get_rotation_matrix(w)

        # save resulting rotated face
        mesh = Mesh(vertices=(T @ G_h)[:3].T, colors=mean_tex, triangles=triangles)
        mesh_to_png("./results/rotation/"+str(w)+".png", mesh)

    return


def facial_landmarks(alpha, delta, w, t):
    """
    Construct facial landmarks from facial geometry latent parameters alpha, delta and object transformation w, t.

    :param alpha: array, 30dim
    :param delta: array, 20dim
    :param w: rotation angles around x,y, z. Given as list [theta_x, theta_y, theta_z].
    :param t: translation in x,y,z space. Given as list [translation_x, translation_y, translation_z]
    :return:
    """
    landmarks_idx = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)

    pca = read_pca_model()
    G = get_face_point_cloud(pca, alpha, delta)[landmarks_idx].T

    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

    # get T matrix
    T = np.eye(4)
    T[:3, :3] = get_rotation_matrix(w)
    T[:3, 3] = t

    # Get V and P matrices
    W = 255
    H = 255

    image_aspect_ratio = W / H
    angle = 10
    near = 300
    far = 2000

    right, left, top, bottom = get_perspective(image_aspect_ratio, angle, near, far)

    V = get_V(right, left, top, bottom)

    P = get_P(near, far, right, left, top, bottom)

    i =  V @ P @ T @ G_h

    # cartesian
    i /= i[3,:]

    # two-dimensional
    return i[:2, :]