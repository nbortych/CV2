
import torch 
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image, ImageSequence
import dlib

from viewport_matrix import get_V
from perspective_projection_matrix import get_perspective, get_P
from morphable_model import get_face_point_cloud, read_pca_model, U, random_face_point_cloud
import matplotlib.pyplot as plt

def rotation_matrix(w, is_numpy=False):
    if is_numpy:
        w = torch.from_numpy(w)

    theta1, theta2, theta3 = w[0], w[1], w[2]
    
    zero = theta1.detach()*0
    one = zero.clone()+1
    
    cosx, sinx, cosy, siny, cosz, sinz = theta1.cos(), theta1.sin(), theta2.cos(), theta2.sin(), theta3.cos(),  theta3.sin()
    
    r_x = torch.stack([one, zero, zero,
                        zero,  cosx, sinx,
                        zero,  -sinx,  cosx]).view( 3, 3)
    
    r_y = torch.stack([cosy, zero,  -siny,
                        zero,  one, zero,
                        siny, zero,  cosy]).view( 3, 3)
    
    r_z = torch.stack([cosz, -sinz, zero,
                        sinz,  cosz, zero,
                        zero, zero,  one]).view( 3, 3)
    
    R = r_x @ r_y @ r_z
    
    if is_numpy:
        R = R.numpy()
    return R
    
def get_P(n, f, t, b, is_numpy = False):
    if is_numpy:
        return np.array([[(2 * n) / (t-b), 0, 0, 0],
                [0, (2 * n) / (t - b), 0, 0],
              [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
              [0, 0, -1, 0]])
    else:
        return torch.Tensor([[(2 * n) / (t-b), 0, 0, 0],
              [0, (2 * n) / (t - b), 0, 0],
              [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
              [0, 0, -1, 0]])



def normalise(landmarks, is_ground = False, values =None):
    
    max_x = torch.max(landmarks[:,0].detach())
    max_y = torch.max(landmarks[:,1].detach())
    min_x = torch.min(landmarks[:,0].detach())
    min_y = torch.min(landmarks[:,1].detach())

    
    scale=torch.sqrt((max_x-min_x).pow(2) + (max_y-min_y).pow(2))
    
    if values!=None:
        length, min_x, min_y = values
    landmarks[:,0] = (landmarks[:,0] - min_x)/scale 
    landmarks[:,1] = (landmarks[:,1] - min_y)/scale
    if is_ground:
        return landmarks, [scale, min_x, min_y]
    return landmarks



def denormalise(estimated_landmarks, target_landmarks, is_numpy = False):
    if is_numpy:
        estimated_landmarks, target_landmarks = torch.from_numpy(estimated_landmarks) ,  torch.form_numpy(target_landmarks)
    landmarks, values = normalise(target_landmarks, is_ground = True)
    
    estimated_landmarks = normalise(estimated_landmarks)
    estimated_landmarks[:,0] = estimated_landmarks[:,0]*values[0]+values[1]
    estimated_landmarks[:,1] = estimated_landmarks[:,1]*values[0]+values[2]
    estimated_landmarks = estimated_landmarks.detach().numpy()
    
    return estimated_landmarks


def get_face_point_cloud_torch(p, alpha, delta): 
    """
    Get face point cloud for given alpha and delta.

    :param p: PCA model received with read_pca_model()
    :param alpha: size 30
    :param delta: size 20
    :return: 3D point cloud of size [num_points x 3]
    """
    G_id = torch.from_numpy(p["mu_id"]) + torch.from_numpy(p["E_id"]) @ ( torch.from_numpy(p["sigma_id"]) * alpha)
    G_ex = torch.from_numpy(p["mu_ex"]) + torch.from_numpy(p["E_ex"]) @ ( torch.from_numpy(p["sigma_ex"]) * delta)
    return (G_id+G_ex).view((-1, 3))





def facial_landmarks_torch(alpha, delta, w, t):
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
    G = get_face_point_cloud_torch(pca, alpha, delta)[landmarks_idx].t()
    G_h = [G , torch.ones(G.shape[1]).view((1, -1))]
    G_h = torch.cat(G_h, dim=0)
    
    # get T matrix
    T = torch.eye(4)
    T[:3, :3] = rotation_matrix(w)#rotation_tensor(w, 1)#get_rotation_matrix_torch(w)  #torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])#
    T[:3, 3] = t
    
    # Get V and P matrices
    W = 172
    H = 162

    image_aspect_ratio = W / H
    angle = 10
    near = .1
    far = 10

    right, left, top, bottom = get_perspective(image_aspect_ratio, angle, near, far)
    
    V = get_V(right, left, top, bottom)

    
    [V] = list(map(torch.from_numpy, [V]))
    V = V.to(dtype = torch.float32)
    n,f, t, b = near, far, top, bottom
    P = torch.Tensor([[(2 * n) / (t-b), 0, 0, 0],
                [0, (2 * n) / (t - b), 0, 0],
              [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
              [0, 0, -1, 0]])
    i =  V @ P @ T @ G_h

    # homo to cartesian
    i = i/i[3,:].clone()

    # two-dimensional
    return i[:2, :].t()



def get_final_landmarks(alpha, delta, w, t, target_landmarks):
    estimated_landmarks = facial_landmarks_torch(alpha, delta, w, t)
    estimated_landmarks = denormalise(estimated_landmarks, target_landmarks)
    return estimated_landmarks