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
from landmarks import detect_landmark
import trimesh
from data_def import PCAModel, Mesh
from mesh_to_png import triangles, mean_tex
from mesh_to_png import mesh_to_png

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

def facial_landmarks_torch(alpha, delta, w, t, LM=True):
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

    if LM:
        G = get_face_point_cloud_torch(pca, alpha, delta)[landmarks_idx].t()
    else:
        G = get_face_point_cloud_torch(pca, alpha, delta).t()

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

def get_ground_truth_landmarks(img, predictor=None):
    if predictor is None:
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    ground_truth = np.array([(point.x, point.y) for point in predictor(img, dets[0]).parts()])

    return ground_truth

def get_final_landmarks(alpha, delta, w, t, target_landmarks):

    estimated_all = facial_landmarks_torch(alpha, delta, w, t, LM=False)
    estimated_all = denormalise(estimated_all, target_landmarks)

    return estimated_all


def find_corresponding_texture(points, image):
    im_height, im_width, _ = image.shape

    new_texture = np.zeros((0, 3))
    out_of_bounds = []

    for i, point in enumerate(points):
        x, y = point[0] + 1e-2, point[1] + 1e-2
        if y < 0 or y > im_height or x < 0 or x > im_width:
            # Save indices of points that are outside the image, such that we may delete those later
            out_of_bounds.append(i)
            continue
        x_low, x_high = int(np.floor(x)), int(np.ceil(x))
        y_low, y_high = int(np.floor(y)), int(np.ceil(y))
        p_x, p_y = (x - x_low) / (x_high - x_low), (y - y_low) / (y_high - y_low)

        color_11, color_12, color_21, color_22 = image[y_low, x_low], image[y_low, x_high], \
                                                 image[y_high, x_low], image[y_high, x_high]

        hori_color_1, hori_color_2 = color_11 * (1 - p_x) + color_12 * p_x, color_21 * (1 - p_x) + color_22 * p_x
        final_color = hori_color_1 * (1 - p_y) + hori_color_2 * p_y

        new_texture = np.append(new_texture, final_color.reshape((1, 3)), axis=0)

    return new_texture
def main():
## def texturing():
    im = Image.open('faces_sparser_sampling.gif')
    frames =     np.array([np.array(frame.copy().convert('RGB').getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],3)
                       for frame in ImageSequence.Iterator(im)])
    image = frames[0] #this is the image
    
    target_landmarks = torch.from_numpy(detect_landmark(image)).to(dtype = torch.float)
    
    alpha, delta, w, t = [i for i in np.load("best_params.npy", allow_pickle=True)]
    
    estimated_all = get_final_landmarks(alpha, delta, w, t, target_landmarks)
    
    #print(estimated_landmarks.shape)
    print(estimated_all.shape)
    
    plt.imshow(image)
    plt.show()
    
    plt.scatter(estimated_all.T[0],estimated_all.T[1])
    plt.imshow(image)
    plt.show()
    
    plt.scatter(estimated_landmarks.T[0], estimated_landmarks.T[1])
    plt.imshow(image)
    plt.show()
    
    tex = find_corresponding_texture(estimated_all, image)
    
    # get 3D point cloud
    p = read_pca_model()
    G = get_face_point_cloud(p, alpha.detach().numpy(), delta.detach().numpy())
    mesh = trimesh.base.Trimesh(vertices=G, faces=triangles, vertex_colors=tex)
    mesh_to_png("mesh.png", mesh)
    t = mesh_to_png("mesh.png", mesh)
    return t