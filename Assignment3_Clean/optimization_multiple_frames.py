from PIL import Image, ImageSequence
import torch
from torch.utils.data import DataLoader, Dataset
import dlib
import numpy as np

from morphable_model import read_pca_model, get_face_point_cloud
from matrices import viewport_matrix, perspective_projection_matrix, rotation_matrix
from texturing import texture

import trimesh
from mesh_to_png import triangles, mean_tex, mesh_to_png
from data_def import Mesh
import matplotlib.pyplot as plt

###### Helpers
def read_pca_model_torch():
    p = read_pca_model()
    for i in p:
        p[i] = torch.from_numpy(p[i])
    return p

def detect_landmarks(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    predicted = predictor(img, detector(img, 1)[0])
    landmarks = np.array([(p.x, p.y) for p in predicted.parts()])
    return torch.from_numpy(landmarks).to(dtype=torch.float)

def loss_function(landmarks, target_landmarks, alpha, delta, l_alpha=1, l_delta=1):
    # both of shape [68,2]
    loss_lan = torch.sum(torch.norm((landmarks - target_landmarks), dim=1).pow(2))
    loss_reg = l_alpha * (alpha.pow(2)).sum() + l_delta * (delta.pow(2)).sum()
    return loss_lan+loss_reg
##################

###### Model
class Landmarks_Fit_Model(torch.nn.Module):

    def __init__(self, image_width, image_height, lm_indices, num_frames):
        super().__init__()

        # initialize variables to optimize
        self.alpha = torch.nn.Parameter(torch.zeros(30), requires_grad=True)
        self.delta = torch.nn.Parameter(torch.zeros(num_frames, 20), requires_grad=True)
        R, t = self.init_R_and_t(num_frames)
        self.R = torch.nn.Parameter(R, requires_grad=True)
        self.t = torch.nn.Parameter(t, requires_grad=True)

        # set V and P matrices (fixed)
        self.V = viewport_matrix(right=image_width, left=0, top=image_height, bottom=0, is_numpy=False)
        self.V[1, 1] = -self.V[1, 1] # 180 degree turn

        self.P = perspective_projection_matrix(image_width, image_height, 300, 2000, is_numpy=False)

        # read facial parameters (fixed)
        self.p = read_pca_model_torch()

        self.lm_indices = lm_indices

    def init_R_and_t(self, num_frames):
        R = torch.zeros(num_frames, 3, 3)
        t = torch.zeros(num_frames, 3)
        for i in range(num_frames):
            R[i] = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
            t[i] = torch.tensor([0.0, 0.0, -400.0])
        return R,t

    def matrix_transformation(self,frame_idx):
        T = torch.eye(4)
        T[:3, :3] = self.R[frame_idx]
        T[:3, 3] = self.t[frame_idx]
        return self.V @ self.P @ T

    def forward(self, frame_idx, only_lm=True):
        """
        Forward pass.
        Aka: compute 2D landmarks with current Variables for input point cloud.

        :param input:  68, 3
        :param target: 68, 2
        :return:
        """
        # calculate current face point cloud
        G = get_face_point_cloud(self.p, self.alpha, self.delta[frame_idx]).view((-1, 3)) # 28588, 3

        # get current landmarks
        if only_lm:
            G = G[self.lm_indices]
        G_lm_h = torch.cat((G, torch.zeros(G.shape[0], 1)), dim=1) # homogeneous

        lm = self.matrix_transformation(frame_idx) @ G_lm_h.t()

        lm /= lm.clone()[3,:] # cartesian
        lm  = lm.clone()[:2, :] # two-dimensional

        return lm.t()
##################

###### Dataset
class Faces(Dataset):
    def __init__(self, path, data_size=300):
        im = Image.open(path)
        frames = np.array([np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1],frame.size[0], 3)
                           for frame in ImageSequence.Iterator(im)])
        self.data = frames[:data_size]
        self.target = []
        for i, img in enumerate(self.data):
            print(i) 
            self.target.append(detect_landmarks(img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.target[idx])
##################


###### OPTIMIZATION FUNCTION
def optimization(num_epochs, path, shape, num_frames, lambda_alpha=45, lambda_delta=15, lr=.128):

    # hyperparams
    lm_indices = np.loadtxt("./models/Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)

    dataset = Faces(path=path, data_size=num_frames)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0)
    assert num_frames==len(dataloader)

    # define optimizer
    image_width, image_height = shape
    lf_model = Landmarks_Fit_Model(image_width, image_height, lm_indices, num_frames)
    optim = torch.optim.Adam(lf_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optim.zero_grad()

        losses = torch.zeros(num_frames)
        for idx, (_, target) in enumerate(dataloader):
            output = lf_model.forward(idx)

            loss = loss_function(output,
                                 target.view(-1,2),
                                 lf_model.alpha,
                                 lf_model.delta[idx],
                                 l_alpha=lambda_alpha,
                                 l_delta=lambda_delta)
            losses[idx]=loss

        loss = torch.sum(losses) # TODO : Or mean ?
        loss.backward()
        optim.step()

        print(f"EPOCH: {epoch} \t LOSS: {loss}")

    return lf_model, dataset.data
#################

###### TASK 4
def test_optimization_single_frame(path='./images/first_frame.png'):
    image = Image.open(path)

    lambda_alpha = 30
    lambda_delta = 0.3
    lr = .128
    num_epochs = 400
    num_frames = 1

    trained_model, frames = optimization(num_epochs=num_epochs,
                                         path=path,
                                         shape=(image.width, image.height),
                                         num_frames=num_frames,
                                         lambda_alpha=lambda_alpha,
                                         lambda_delta=lambda_delta,
                                         lr=lr)

    pca = trained_model.p

    # look at 3D visualization of the estimated face shape
    points_3d = get_face_point_cloud(pca, trained_model.alpha, trained_model.delta[0]).view((-1, 3))  # 28588, 3

    # obtain texture from mask on image

    points_2d = trained_model.forward(0,only_lm=False).detach().numpy()  # 28588, 2
    tex = np.array(texture(np.array(image), points_2d))
    
    #mesh = trimesh.base.Trimesh(vertices=points_3d.detach().numpy(), faces=triangles, vertex_colors=tex)
    #mesh.show()

    # show estimated landmarks
    landmarks = trained_model.forward(0)
    landmarks = landmarks.detach().numpy().T
    plt.scatter(landmarks[0], landmarks[1])
    plt.imshow(np.array(frames[0]))
    plt.axis('off')
    plt.savefig(f"./results/optimization/landmarks.png",
                bbox_inches='tight', pad_inches=0, dpi=100)
    
    # show from different angles
    G = points_3d.detach().numpy().T
    G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)

    for w in [[0, 0,0], [0,-30,0], [0,-45,0], [0,-90,0]]:
        w = np.array(w)
        # get T matrix for only rotation
        T = np.eye(4)
        T[:3, :3] = rotation_matrix(w, is_numpy=True)
        # save resulting rotated face
        mesh = Mesh(vertices=(T @ G_h)[:3].T, colors=tex, triangles=triangles)
        mesh_to_png(f"./results/optimization/single/tex_{w}.png", mesh,z_camera_translation=280)
#################

###### TASK 6
def test_optimization_multiple_images():
    image = Image.open('./images/faces_sparser_sampling.gif')

    lambda_alpha = 50
    lambda_delta = 0.3
    lr = .128
    num_epochs = 300
    num_frames = 100

    trained_model, frames = optimization(num_epochs=num_epochs,
                                         path='./images/faces_sparser_sampling.gif',
                                         shape=(image.width, image.height),
                                         num_frames=num_frames,
                                         lambda_alpha=lambda_alpha,
                                         lambda_delta=lambda_delta,
                                         lr=lr)

    pca = trained_model.p

    for frame_idx,frame in enumerate(frames):
        # obtain texture from mask on image
        points_2d = trained_model.forward(frame_idx, only_lm=False).detach().numpy()  # 28588, 2
        tex = np.array(texture(np.array(frame), points_2d))

        # look at 3D visualization of the estimated face shape
        points_3d = get_face_point_cloud(pca, trained_model.alpha, trained_model.delta[frame_idx]).view((-1, 3))
        """    
        mesh = trimesh.base.Trimesh(vertices=points_3d.detach().numpy(), faces=triangles, vertex_colors=tex)
        mesh.show()
        """
        
        
        # get T matrix for only rotation
        T = np.eye(4)
        T[:3, :3] = rotation_matrix(np.array([0,0,0]), is_numpy=True)
        # save resulting rotated face
        G = points_3d.detach().numpy().T
        G_h = np.append(G, np.ones(G.shape[1]).reshape((1, -1)), axis=0)
        mesh = Mesh(vertices=(T @ G_h)[:3].T, colors=tex, triangles=triangles)
        mesh_to_png(f"./results/optimization/multiple/shapes/{frame_idx}.png", mesh, z_camera_translation=280)

        # show estimated landmarks
        landmarks = trained_model.forward(frame_idx)
        landmarks = landmarks.detach().numpy().T
        plt.figure(figsize=(600 / 100, 600 / 100), dpi=100)
        #plt.scatter(landmarks[0], landmarks[1])
        plt.imshow(np.array(frame))
        plt.axis('off')
        plt.savefig(f"./results/optimization/multiple/landmarks/{frame_idx}.png",
                    bbox_inches='tight', pad_inches=0, dpi=100)

#################

test_optimization_multiple_images()
