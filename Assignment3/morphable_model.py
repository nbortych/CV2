import os
import numpy as np
import trimesh
import h5py
from data_def import Mesh
from mesh_to_png import triangles, mean_tex, mesh_to_png

def U(i=1):
    """ 
    Note: The higher you choose i, the further away the sampled face is from the mediocre face.
    i=1 gives almost always the same face.
    i=3 gives very expressive faces of various shape.
    ...
    """
    return np.random.uniform(-i, i)

def read_pca_model(num_pc_id=30, num_pc_ex=20):
    bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')

    # facial identity
    mu_id  = np.array(bfm['shape/model/mean']) # (3N,)
    sigma_id = np.sqrt(np.array(bfm['shape/model/pcaVariance']))[:num_pc_id] # (199,)
    E_id = np.array(bfm['shape/model/pcaBasis'])[:,:num_pc_id] # 3N,199)

    # facial expression 
    mu_ex = np.array(bfm['expression/model/mean']) # (3N)
    sigma_ex = np.sqrt(np.array(bfm['shape/model/pcaVariance']))[:num_pc_ex] # (100,)
    E_ex = np.array(bfm['shape/model/pcaBasis'])[:,:num_pc_ex] # (3N, 100)

    return {"mu_id": mu_id, "sigma_id": sigma_id, "E_id": E_id,
            "mu_ex": mu_ex, "sigma_ex": sigma_ex, "E_ex": E_ex}

def sample_face(p, alpha, delta):
    """
    p : PCA model received with read_pca_model()
    """
    G_id = p["mu_id"] + p["E_id"] @ ( p["sigma_id"] * alpha)
    G_ex = p["mu_ex"] + p["E_ex"] @ ( p["sigma_ex"] * delta)
    return G_id+G_ex

def sample_face_pointclouds(num_samples=24):
    # read pca model from files
    pca_model = read_pca_model()

    for i in range(num_samples):
        # sample new face with given formula
        alpha = U(i=1)
        delta = U(i=1)
        f_pc = sample_face(pca_model, alpha, delta).reshape((-1, 3)) # (3N, )= (85764,)
        
        # show mesh:
        # mesh = trimesh.base.Trimesh(vertices=f_pc, faces=triangles, vertex_colors=mean_tex)
        # mesh.show()
        
        # save mesh
        mesh = Mesh(vertices=f_pc, colors=mean_tex, triangles=triangles)
        mesh_to_png(f"./res_morphable_model/{str(i)}.png", mesh)
        
sample_face_pointclouds()
