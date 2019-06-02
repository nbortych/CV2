import os
import numpy as np
import trimesh
import h5py
from data_def import Mesh
from mesh_to_png import triangles, mean_tex, mesh_to_png

def U(size):
    return np.random.uniform(-1, 1, size=size)

def read_pca_model(num_pc_id=30, num_pc_ex=20):
    bfm = h5py.File("./models/model2017-1_face12_nomouth.h5", 'r')

    # facial identity
    mu_id  = np.array(bfm['shape/model/mean']) # (3N,)
    sigma_id = np.sqrt(np.array(bfm['shape/model/pcaVariance']))[:num_pc_id] # (199,)
    E_id = np.array(bfm['shape/model/pcaBasis'])[:,:num_pc_id] # 3N,199)

    # facial expression 
    mu_ex = np.array(bfm['expression/model/mean']) # (3N)
    sigma_ex = np.sqrt(np.array(bfm['expression/model/pcaVariance']))[:num_pc_ex] # (100,)
    E_ex = np.array(bfm['expression/model/pcaBasis'])[:,:num_pc_ex] # (3N, 100)

    return {"mu_id": mu_id, "sigma_id": sigma_id, "E_id": E_id,
            "mu_ex": mu_ex, "sigma_ex": sigma_ex, "E_ex": E_ex}

def get_face_point_cloud(p, alpha, delta):
    """
    Get face point cloud for given alpha and delta.

    :param p: PCA model received with read_pca_model()
    :param alpha: size 30
    :param delta: size 20
    :return: 3D point cloud of size [num_points x 3]
    """
    G_id = p["mu_id"] + p["E_id"] @ ( p["sigma_id"] * alpha)
    G_ex = p["mu_ex"] + p["E_ex"] @ ( p["sigma_ex"] * delta)
    return G_id+G_ex

def random_face_point_cloud(p):
    """
    Sample random new face point cloud.
    :param p:
    :return:
    """
    point_cloud = get_face_point_cloud(p, U(30), U(20))
    return point_cloud.reshape((-1, 3))

def generate_face_images(num_samples=24):
    # read pca model from files
    pca_model = read_pca_model()

    for i in range(num_samples):
        # sample new face with given formula
        f_pc = random_face_point_cloud(pca_model) # (3N, )= (85764,)

        # show mesh:
        # mesh = trimesh.base.Trimesh(vertices=f_pc, faces=triangles, vertex_colors=mean_tex)
        # mesh.show()

        # save mesh
        mesh = Mesh(vertices=f_pc, colors=mean_tex, triangles=triangles)
        mesh_to_png(f"./results/morphable_model/{str(i)}.png", mesh, width=400,z_camera_translation=280)

    return