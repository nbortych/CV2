from PIL import Image, ImageSequence
import torch
from torch.utils.data import DataLoader, Dataset
import dlib
import numpy as np

from morphable_model import read_pca_model, get_face_point_cloud
from matrices import viewport_matrix, perspective_projection_matrix,rotation_matrix

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

    def __init__(self, image_width, image_height, lm_indices):
        super().__init__()

        # initialize variables to optimize
        self.alpha = torch.nn.Parameter(torch.zeros(30), requires_grad=True)
        self.delta = torch.nn.Parameter(torch.zeros(20), requires_grad=True)
        self.R = torch.nn.Parameter(torch.tensor([[1.0,  0.0, 0.0],
                                                  [0.0, -1.0, 0.0],
                                                  [0.0,  0.0, 1.0]]), requires_grad=True)
        self.t = torch.nn.Parameter(torch.tensor([0.0, 0.0, -400.0]), requires_grad=True)

        # set V and P matrices (fixed)
        self.V = viewport_matrix(right=image_width, left=0, top=image_height, bottom=0, is_numpy=False)
        self.V[1, 1] = -self.V[1, 1] # 180 degree turn

        self.P = perspective_projection_matrix(image_width, image_height, 300, 2000, is_numpy=False)

        # read facial parameters (fixed)
        self.p = read_pca_model_torch()

        self.lm_indices = lm_indices

    def matrix_transformation(self):
        T = torch.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return self.V @ self.P @ T

    def forward(self, only_lm=True):
        """
        Forward pass.
        Aka: compute 2D landmarks with current Variables for input point cloud.

        :param input:  68, 3
        :param target: 68, 2
        :return:
        """
        # calculate current face point cloud
        G = get_face_point_cloud(self.p, self.alpha, self.delta).view((-1, 3)) # 28588, 3

        # get current landmarks
        if only_lm:
            G = G[self.lm_indices]
        G_lm_h = torch.cat((G, torch.zeros(G.shape[0], 1)), dim=1) # homogeneous

        lm = self.matrix_transformation() @ G_lm_h.t()

        lm /= lm.clone()[3,:] # cartesian
        lm  = lm.clone()[:2, :] # two-dimensional

        return lm.t()
##################



###### Task 4
def optimization_one_image(num_steps, image, lambda_alpha=1, lambda_delta=1, lr=.128):

    lm_indices = np.loadtxt("./models/Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)

    # define optimizer
    lf_model= Landmarks_Fit_Model(image.width, image.height, lm_indices)
    optim = torch.optim.Adam(lf_model.parameters(), lr=lr)

    # get ground truth landmarks
    target = detect_landmarks(np.array(image))

    # train
    for step in range(num_steps):

        optim.zero_grad()#
        output = lf_model.forward() # 68 , 2
        loss = loss_function(output,
                             target,
                             lf_model.alpha,
                             lf_model.delta,
                             l_alpha = lambda_alpha,
                             l_delta = lambda_delta)
        loss.backward()
        optim.step()

        if step%10==0:
            print(f"STEP: {step} \t LOSS: {loss}")

    return lf_model

def test_optimization_one_image():
    image = Image.open('./images/first_frame.png')

    # TODO: Report what happens if we set different hyperparameters here
    lambda_alpha=45
    lambda_delta=15
    lr=.128

    trained_model = optimization_one_image(300,
                                           image,
                                           lambda_alpha=lambda_alpha,
                                           lambda_delta=lambda_delta,
                                           lr=lr)

    pca = trained_model.p

    # show estimated landmarks
    landmarks = trained_model.forward()
    landmarks = landmarks.detach().numpy().T
    plt.scatter(landmarks[0], landmarks[1])
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.savefig(f"./results/optimization/landmarks_{lambda_alpha}_{lambda_delta}_{lr}.png")

    # look at 3D visualization of the estimated face shape
    points_3d = get_face_point_cloud(pca, trained_model.alpha, trained_model.delta).view((-1, 3))  # 28588, 3
    mesh = trimesh.base.Trimesh(vertices=points_3d.detach().numpy(),faces=triangles,vertex_colors=mean_tex)
    mesh.show()

    # alternative: save it as png & look at it then
    mesh = Mesh(vertices=points_3d.detach().numpy(), colors=mean_tex, triangles=triangles)
    mesh_to_png(f"./results/optimization/shape_{lambda_alpha}_{lambda_delta}_{lr}.png", mesh)
##################



###### Task 6: TODO: Work in progress

class Faces(Dataset):
    def __init__(self, path, data_size=300):
        im = Image.open(path)
        frames = np.array([np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1],frame.size[0], 3)
                           for frame in ImageSequence.Iterator(im)])
        self.data = frames[:data_size]
        self.target = [detect_landmarks(img) for img in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.target[idx])

def optimization_multiple_images(num_epochs, path, batch_size, shape, lambda_alpha=45, lambda_delta=15, lr=.128):

    # hyperparams
    lm_indices = np.loadtxt("./models/Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)


    dataset = Faces(path=path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    # define optimizer
    image_width, image_height = shape
    lf_model = Landmarks_Fit_Model(image_width, image_height, lm_indices)
    optim = torch.optim.Adam(lf_model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        for idx, (_, target_batch) in enumerate(dataloader):

            # get mean loss for batch
            losses = []
            for target in target_batch:
                optim.zero_grad()
                output = lf_model.forward()
                loss = loss_function(output,
                                     target,
                                     lf_model.alpha,
                                     lf_model.delta,
                                     l_alpha=lambda_alpha,
                                     l_delta=lambda_delta)
                losses.append(loss)

            mean_loss = torch.from_numpy(np.mean(losses))
            mean_loss.backward()
            optim.step()

            overall_step = idx + len(dataloader)*epoch

            if overall_step%10:
                mean_loss = np.mean(losses)
                losses = []
                print(f"STEP: {overall_step} \t LOSS: {mean_loss}")

    #  TODO : ....

    return lf_model


def testoptimization_multiple_images():
    image = Image.open('./images/first_frame.png')

    lambda_alpha=45
    lambda_delta=15
    lr=.128

    trained_model = optimization_multiple_images(num_epochs = 5,
                                                 path='./images/faces_sparser_sampling.gif',
                                                 batch_size=10,
                                                 shape = (image.width, image.height),
                                                 lambda_alpha=lambda_alpha,
                                                 lambda_delta=lambda_delta,
                                                 lr=lr)
    #  TODO : ....
##################

