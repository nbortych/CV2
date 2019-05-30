import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import h5py
import argparse
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from data_def import Mesh
import trimesh
import pyrender
import h5py
from mpl_toolkits import mplot3d
from skimage.io import imsave

def mesh_to_png(file_name, mesh, width=640, height=480, z_camera_translation=400):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, wireframe=False)

    # compose scene
    scene = pyrender.Scene(ambient_light=np.array([1.7, 1.7, 1.7, 1.0]), bg_color=[255, 255, 255])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    # Added camera translated z_camera_translation in the 0z direction w.r.t. the origin
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, z_camera_translation],
                            [0, 0, 0, 1]])

    # render scene
    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)

    imsave(file_name, color)

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [3, 3]
    """
    x, y, z = angle[0], angle[1], angle[2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones]).reshape(-1, 3, 3,).squeeze(0)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=0).reshape(-1, 3, 3,).squeeze(0)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=0).reshape(-1, 3, 3,).squeeze(0)

    rotMat = xmat @ ymat @ zmat
    return rotMat

class Headcrab(nn.Module):
    def __init__(self):
        super(Headcrab, self).__init__()

        # Storing the face parameters in Pytorch
        bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')
        self.mu_id = torch.from_numpy(np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3)))
        self.mu_exp = torch.from_numpy(np.asarray(bfm['expression/model/mean'], dtype=np.float32).reshape((-1, 3)))

        E_id = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32).reshape((-1, 199))
        E_exp = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32).reshape((-1, 100))
        self.E_id = torch.from_numpy(E_id)
        self.E_exp = torch.from_numpy(E_exp)

        var_id = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)
        var_exp = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)

        self.var_id = torch.from_numpy(var_id)
        self.var_exp = torch.from_numpy(var_exp)

        self.E_id = torch.from_numpy(E_id[:, :30].reshape(-1, 3, 30))
        self.E_exp = torch.from_numpy(E_exp[:, :20].reshape(-1, 3, 20))

        self.sigma_id = torch.from_numpy(np.sqrt(var_id[:30]))
        self.sigma_exp = torch.from_numpy(np.sqrt(var_exp[:20]))

        # Storing landmark indices
        indices_file = open("Landmarks68_model2017-1_face12_nomouth.anl")
        indices = indices_file.read().splitlines()
        self.landmark_indices = list(map(int, indices))

        # Using it to get input size
        # input_size = len(self.landmark_indices)

        alpha_size = 30
        delta_size = 20

        t = np.array([0,0,-400])
        omega = np.array([np.pi, 0, 0])

        self.alpha = nn.Parameter(torch.FloatTensor(alpha_size).uniform_(-1, 1))
        self.delta = nn.Parameter(torch.FloatTensor(delta_size).uniform_(-1, 1))
        self.omega = nn.Parameter(torch.from_numpy(omega).float())
        self.t = nn.Parameter(torch.from_numpy(t).float())

    def forward(self):

        alpha = self.alpha
        delta = self.delta
        omega = self.omega
        t = self.t


        # Step 2: Calculating face geometry
        geometry = self.mu_id + self.E_id @ (alpha * self.sigma_id) + self.mu_exp + self.E_exp @ (delta * self.sigma_exp)
        R = euler2mat(omega)
        geometry_transformed = geometry @ R + t

        # Step 4: Getting landmark_coordinates
        landmark_guess = geometry_transformed[self.landmark_indices, 0:2]

        # normalize coordinates to [0,1] https://stackoverflow.com/questions/3862096/2d-coordinate-normalization
        maxX = torch.max(landmark_guess[:,0])
        minX = torch.min(landmark_guess[:,0])
        maxY = torch.max(landmark_guess[:,1])
        minY = torch.min(landmark_guess[:,1])

        length_of_diagonal=torch.sqrt((maxX-minX)*(maxX-minX) + (maxY-minY)*(maxY-minY))
        normalized_landmark_guess = torch.Tensor(landmark_guess.shape)
        normalized_landmark_guess[:,0] = (landmark_guess[:,0] - minX)/(length_of_diagonal)
        normalized_landmark_guess[:,1] = (landmark_guess[:,1] - minY)/(length_of_diagonal)

        return normalized_landmark_guess, (length_of_diagonal, minX, minY)

def main():
    model = Headcrab()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


    file = open("faces/emma_points.pkl",'rb')
    points = pickle.load(file)
    points = torch.from_numpy(points).float()

    # normalize coordinates to [0,1] https://stackoverflow.com/questions/3862096/2d-coordinate-normalization
    maxX = torch.max(points[:,0])
    minX = torch.min(points[:,0])
    maxY = torch.max(points[:,1])
    minY = torch.min(points[:,1])

    length_of_diagonal = torch.sqrt((maxX-minX)*(maxX-minX) + (maxY-minY)*(maxY-minY))
    points[:,0] = (points[:,0] - minX)/(length_of_diagonal)
    points[:,1] = (points[:,1] - minY)/(length_of_diagonal)


    for i in range(100):
        if i % 20 == 0:
            scheduler.step()

        optimizer.zero_grad()

        landmark_guess, normalization_values = model.forward()

        lambda_alpha = 0.001
        lambda_delta = 0.001

        L_lan = torch.norm(landmark_guess - points).pow(2).sum()
        L_reg = (lambda_alpha * model.alpha.pow(2)).sum() + (lambda_delta * model.delta.pow(2)).sum()
        loss = L_lan + L_reg
        print(loss.item())

        loss.backward()
        optimizer.step()


    fp = open("faces/emma_latent.pkl","wb")
    pickle.dump({"alpha": model.alpha.detach().numpy(), "delta": model.delta.detach().numpy(), "omega": model.omega.detach().numpy(), "t": model.t.detach().numpy(), "length_of_diagonal": length_of_diagonal, "minX": minX, "minY": minY, "length_of_diagonal_landmarks":  normalization_values[0].detach().numpy(), "minX_landmarks":  normalization_values[1].detach().numpy(), "minY_landmarks":  normalization_values[2].detach().numpy()}, fp)
    fp.close()

    # make resulting landmarks
    points[:,0] = points[:,0]*length_of_diagonal + minX
    points[:,1] = points[:,1]*length_of_diagonal + minY

    landmark_guess[:,0] = landmark_guess[:,0]*length_of_diagonal + minX
    landmark_guess[:,1] = landmark_guess[:,1]*length_of_diagonal + minY

    image = mpimg.imread("faces/emma.jpg")
    plt.imshow(image)
    # put a red dot, size 40, at 2 locations:
    plt.scatter(points[:,0],points[:,1], c='g', s=20)
    plt.scatter(landmark_guess[:,0].detach().numpy(), landmark_guess[:,1].detach().numpy(), c='r', s=20)
    plt.show()


    G = model.mu_id + model.E_id @ (model.alpha * model.sigma_id) + model.mu_exp + model.E_exp @ (model.delta * model.sigma_exp)
    # show face geometry
    bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')
    mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

    mesh = Mesh(G.detach().numpy(), mean_tex, triangles)

    mesh_to_png("faces/emma_3d.png", mesh)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    args = parser.parse_args()

    main()
