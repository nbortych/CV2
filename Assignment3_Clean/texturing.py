import numpy as np
from PIL import Image

from optimization import optimization_one_image,read_pca_model_torch
from morphable_model import get_face_point_cloud
from matrices import rotation_matrix

import matplotlib.pyplot as plt
import trimesh
from data_def import Mesh
from mesh_to_png import triangles, mesh_to_png


def texture(image, points_on_image):

    def point_on_image(image, point):
        if 0 < point[0] < image.width-2 and 0 < point[1] < image.height-2:
            return True
        return False

    tex = []
    image_array = np.array(image)

    for p in points_on_image:

        if point_on_image(image,p):

            # Following https://en.wikipedia.org/wiki/Bilinear_interpolation

            # read pixel values from image
            x, y = p[0], p[1]
            x_1, y_1 = int(x), int(y)
            x_2, y_2 = x_1+1, y_1+1

            f_Q_11 = image_array[y_1, x_1, :]
            f_Q_12 = image_array[y_2, x_1, :]
            f_Q_21 = image_array[y_1, x_2, :]
            f_Q_22 = image_array[y_2, x_2, :]

            # linear interpolation in the x - direction
            f_x_y1 = (x_2-x) * f_Q_11 + (x-x_1) * f_Q_21
            f_x_y2 = (x_2-x) * f_Q_12 + (x-x_1) * f_Q_22

            # linear interpolation in the y - direction
            f_x_y = (y_2-y) * f_x_y1 + (y-y_1) * f_x_y2

            tex.append(f_x_y)

        else:
            tex.append(np.zeros(3))

    return tex


def texturing():
    # Task 5
    image = Image.open('./images/first_frame.png')

    trained_model = optimization_one_image(300,
                                           image,
                                           lambda_alpha=45,
                                           lambda_delta=15,
                                           lr=.128)

    pca = trained_model.p

    # show estimated landmarks
    landmarks = trained_model.forward()
    landmarks = landmarks.detach().numpy().T
    plt.scatter(landmarks[0], landmarks[1])
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()

    # show estimated total mask
    points_3d = get_face_point_cloud(pca, trained_model.alpha, trained_model.delta).view((-1, 3))# 28588, 3
    points_2d = trained_model.forward(only_lm=False).detach().numpy() # 28588, 2

    plt.scatter(points_2d.T[0], points_2d.T[1])
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()

    # obtain texture from mask on image
    tex = np.array(texture(image, points_2d))

    # look at 3D visualization
    # mesh = trimesh.base.Trimesh(vertices=points_3d.detach().numpy(),faces=triangles,vertex_colors=new_texture)
    # mesh.show()

    # show from different angles
    G = points_3d.detach().numpy().T
    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

    for w in [[0,10,0], [0,0,0], [0,-10,0]]:
        w = np.array(w)
        # get T matrix for only rotation
        T = np.eye(4)
        T[:3, :3] = rotation_matrix(w, is_numpy=True)

        # save resulting rotated face
        mesh = Mesh(vertices=(T @ G_h)[:3].T, colors=tex, triangles=triangles)
        mesh_to_png(f"./results/texturing/tex_{w}.png", mesh)

    #mesh = Mesh(vertices=, colors=tex, triangles=triangles)
    #mesh_to_png("./results/texturing.png", mesh)

    return
