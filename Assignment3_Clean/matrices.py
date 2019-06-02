import torch
import numpy as np

def rotation_matrix(w, is_numpy=False):
    """
    Get rotation matrix for angles theta = [theta_x, theta_y, theta_z].
    Theta is represented in degrees originally.
    """
    w = np.array([np.deg2rad(i) for i in w])
    w = torch.from_numpy(w).to(dtype = torch.float)

    theta1, theta2, theta3 = w[0], w[1], w[2]

    zero = theta1.detach()*0
    one = zero.clone()+1

    cosx, sinx, cosy, siny, cosz, sinz = theta1.cos(), theta1.sin(), theta2.cos(), theta2.sin(), theta3.cos(),  theta3.sin()

    r_x = torch.stack([one, zero, zero,
                        zero,  cosx, sinx,
                        zero,  -sinx,  cosx]).view( 3, 3)

    r_y = torch.stack([cosy, zero,  siny,
                        zero,  one, zero,
                        -siny, zero,  cosy]).view( 3, 3)

    r_z = torch.stack([cosz, -sinz, zero,
                        sinz,  cosz, zero,
                        zero, zero,  one]).view( 3, 3)

    R = r_x @ r_y @ r_z

    if is_numpy:
        R = R.numpy()
    return R


def perspective_projection_matrix(image_width, image_height, near, far, fovy=0.5, is_numpy=False):
    image_aspect_ratio = image_width / image_height

    scale = np.tan(fovy) * near

    right = image_aspect_ratio * scale
    left = -right
    top = scale
    bottom = -scale

    x1 = (2 * near) / (right - left)
    x2 = (right + left) / (right - left)
    x3 = (2 * near) / (top - bottom)
    x4 = (top + bottom) / (top - bottom)
    x5 = -(far + near) / (far - near)
    x6 = -(2 * far * near) / (far - near)

    if is_numpy:
        return np.array([[x1,   0,  x2,  0],
                         [ 0,  x3,  x4,  0],
                         [ 0,   0,  x5, x6],
                         [ 0,   0,  -1,  0]])
    else:
        return torch.Tensor([[x1,   0,  x2,  0],
                             [ 0,  x3,  x4,  0],
                             [ 0,   0,  x5, x6],
                             [ 0,   0,  -1,  0]]).to(dtype = torch.float32)

def viewport_matrix(right, left, top, bottom, is_numpy=True):
    x1 = (right - left)/2
    x2 = (right + left)/2
    y1 = (top - bottom)/2
    y2 = (top + bottom)/2

    V = np.array([[x1,  0,   0,  x2],
                  [ 0, y1,   0,  y2],
                  [ 0,  0, 0.5, 0.5],
                  [ 0,  0,   0,   1]])

    if not is_numpy:
        return torch.from_numpy(V).to(dtype = torch.float32)

    return V

