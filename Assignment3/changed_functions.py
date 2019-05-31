import torch 

def rotation_matrix(w, is_numpy=True):
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
    
def get_P(n, f, t, b, is_numpy = True):
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



def denormalise(estimated_landmarks, target_landmarks, is_numpy = True):
    if is_numpy:
        estimated_landmarks, target_landmarks = torch.from_numpy(estimated_landmarks) ,  torch.form_numpy(target_landmarks)
    landmarks, values = normalise(target_landmarks, is_ground = True)
    
    estimated_landmarks = normalise(estimated_landmarks)
    estimated_landmarks[:,0] = estimated_landmarks[:,0]*values[0]+values[1]
    estimated_landmarks[:,1] = estimated_landmarks[:,1]*values[0]+values[2]
    estimated_landmarks = estimated_landmarks.detach().numpy()
    
    return estimated_landmarks