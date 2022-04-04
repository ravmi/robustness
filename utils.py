import torch
import numpy as np


def net_to_img(data):
    assert data.ndim == 3
    if torch.is_tensor(data):
        return data.permute(1, 2, 0)
    else:
        return data.transpose(1, 2, 0)


def img_to_net(data):
    assert data.ndim == 3
    if torch.is_tensor(data):
        return data.permute(2, 0, 1)
    else:
        return data.transpose(2, 0, 1)

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def unnormalize(data):
    #mean = torch.tensor([8.55112426, 187.17984377, 3.40404531]).reshape(3, 1, 1)
    #std =  torch.tensor([43.36357464, 19.35027745, 17.2615793]).reshape(3, 1, 1)
    mean = np.array([8.55112426, 187.17984377, 3.40404531]).reshape(3, 1, 1)
    std =  np.array([43.36357464, 19.35027745, 17.2615793]).reshape(3, 1, 1)
    return (data * std) + mean

