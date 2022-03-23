import numpy as np
import os
from PIL import Image
import math
import torch
import torch.nn.functional as F
from utils import img_to_net

import sys
import numpy
from torchvision.transforms import Normalize
numpy.set_printoptions(threshold=sys.maxsize)

MIN_ID = 0
MAX_ID = 1100

MIN_AUGMENTED = 0
MAX_AUGMENTED = 9

RESULTS = "/Users/rafal.michaluk/datasets/results"
AUGMENTED = "/Users/rafal.michaluk/datasets/augmented"
torch.set_printoptions(linewidth=20000)



class BrickDataset(torch.utils.data.Dataset):
    def __init__(self, root=RESULTS):
        self.root = root

    def __len__(self):
        return 1100

    def __getitem__(self, i):
        depth = np.load(os.path.join(RESULTS, f"{i}.npy"))
        depth = torch.tensor(depth)
        depth = torch.unsqueeze(depth, 0)

        img = Image.open(os.path.join(RESULTS, f"{i}-in.png"))
        inp = torch.tensor(np.asarray(img)[:, :, :3] * 1.0)
        inp = img_to_net(inp)

        inp = Normalize(mean=[110.38053718566894, 107.84343673706054, 109.98833946228028], std=[68.15320649910386,
            69.23072564755397, 70.3886151478994])(inp)
        depth = Normalize(mean=[0.001250115736038424], std=[0.007071534325436761])(depth)

        x = torch.cat((inp, depth), 0)


        img = Image.open(os.path.join(RESULTS, f"{i}-out.png")).convert("L")
        y = (np.asarray(img) > 128)
        x, y = torch.tensor(x), torch.tensor(y)

        y = y.long()
        y = F.one_hot(y, num_classes=2).permute(2, 0, 1)

        return x.float(), y.float()





def check_4_channel():
    """Checking if fourth channel in input images is useless (it is)"""
    total_min = np.inf
    total_max = -np.inf
    for i in range(MIN_ID, MAX_ID + 1):
        img = Image.open(os.path.join(RESULTS, f"{i}-in.png"))
        ar = np.asarray(img)
        total_min = min((ar[:, :, 3]).min(), total_min)
        total_max = max((ar[:, :, 3]).max(), total_max)
    print(total_min, total_max)

def check_output():
    """checking if input images after grayscaling have only two possible values"""
    # 30 and 215
    total_vals = set()
    for i in range(MIN_ID, MAX_ID + 1):
        img = Image.open(os.path.join(RESULTS, f"{i}-out.png"))
        img2 = img.convert("L")
        ar = np.asarray(img2)
        arl = ar.reshape(-1).tolist()
        total_vals |= set(arl)
    print(total_vals)


def load_sample(i):
    depth = np.load(os.path.join(RESULTS, f"{i}.npy"))
    img = Image.open(os.path.join(RESULTS, f"{i}-in.png"))
    x = np.asarray(img) * 1.0
    x[:, :, 3] = depth
    x = img_to_net(x)

    img = Image.open(os.path.join(RESULTS, f"{i}-out.png")).convert("L")
    y = (np.asarray(img) > 128)
    x, y = torch.tensor(x), torch.tensor(y)


    y = y.long()
    y = F.one_hot(y, num_classes=2).permute(2, 0, 1)


    return x, y



    y_good, y_bad = list(), list()
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j] == 1.:
                y_good.append((i, j))
            else:
                y_bad.append((i, j))

    return y, y_good, y_bad

