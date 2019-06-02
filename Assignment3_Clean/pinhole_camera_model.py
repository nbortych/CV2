import numpy as np

from morphable_model import read_pca_model, random_face_point_cloud, get_face_point_cloud
from matrices  import rotation_matrix, perspective_projection_matrix, viewport_matrix

from data_def import Mesh
from mesh_to_png import triangles, mean_tex, mesh_to_png
import math

def rotate_face(angles):
    """
    Task 3.1
    :param angles: list of angles. each angle has three entries [theta_x, theta_y, theta_z]
    :return:
    """
    # sample face
    pca = read_pca_model()
    G = random_face_point_cloud(pca).T

    # transform to homogeneous coordinates
    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

    for w in angles:
        w = np.array(w)
        # get T matrix for only rotation
        T = np.eye(4)
        T[:3, :3] = rotation_matrix(w, is_numpy=True)

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
    landmarks_idx = np.loadtxt("./models/Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)

    pca = read_pca_model()
    G = get_face_point_cloud(pca, alpha, delta).reshape((-1, 3))[landmarks_idx].T

    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

    # get T matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(w, is_numpy=True)
    T[:3, 3] = t

    # Get V and P matrices
    W = H = 255

    # angle 10
    P = perspective_projection_matrix(W, H, 300, 2000, is_numpy=True)

    V = viewport_matrix(right=W, left=0, top=H, bottom=0, is_numpy=True)

    i =  V @ P @ T @ G_h

    # cartesian
    i /= i[3,:]

    # two-dimensional
    return i[:2, :]