import os
<<<<<<< Updated upstream
from skimage.io import imsave
=======
>>>>>>> Stashed changes
import numpy as np
import trimesh
import pyrender

import h5py

from data_def import PCAModel, Mesh

bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')

mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))

triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

<<<<<<< Updated upstream
def mesh_to_png(file_name, mesh, width=640, height=480, z_camera_translation=400):
=======
def mesh_to_png(file_name, mesh):
>>>>>>> Stashed changes
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)

<<<<<<< Updated upstream
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, wireframe=False)

    # compose scene
    scene = pyrender.Scene(ambient_light=np.array([1.7, 1.7, 1.7, 1.0]), bg_color=[255, 255, 255])
    camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    # Added camera translated z_camera_translation in the 0z direction w.r.t. the origin
    scene.add(camera, pose=[[ 1,  0,  0,  0],
                            [ 0,  1,  0,  0],
                            [ 0,  0,  1,  z_camera_translation],
                            [ 0,  0,  0,  1]])

    # render scene
    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)

    imsave(file_name, color)


if __name__ == '__main__':
    mesh = Mesh(mean_shape, mean_tex, triangles)
    mesh_to_png("debug.png", mesh)
    mesh_to_png("debug1234.png", mesh)
=======
    png = mesh.scene().save_image()
    with open(file_name, 'wb') as f:
        f.write(png)

if __name__ == '__main__':
    mesh = Mesh(mean_shape, mean_tex, triangles)
    mesh_to_png("debug.png", mesh)
>>>>>>> Stashed changes
