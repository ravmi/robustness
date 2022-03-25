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
BOARD_COUNT = 1100

MIN_AUGMENTED = 0
MAX_AUGMENTED = 9
AUGMENTED_COUNT = 10

RESULTS = "datasets/results"
AUGMENTED = "datasets/augmented"


class Experiments():
    @staticmethod
    def mean_std_img():
        inps = list()
        for i in range(BOARD_COUNT):
            img = Image.open(os.path.join(RESULTS, f"{i}-in.png"))
            inp = np.asarray(img)[:, :, :3]

            inps.append(inp.reshape(-1, 3))
        inps = np.concatenate(inps, axis=0)

        return np.mean(inps, axis=0), np.std(inps, axis=0)

    @staticmethod
    def mean_std_depth():
        inps = list()
        for i in range(BOARD_COUNT):
            inp = np.load(os.path.join(RESULTS, f"{i}.npy"))
            inps.append(inp.reshape(-1))
        inps = np.concatenate(inps)

        return np.mean(inps), np.std(inps)


    @staticmethod
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

    @staticmethod
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

        mean = [8.55112426, 187.17984377, 3.40404531]
        std =  [43.36357464, 19.35027745, 17.2615793]
        inp = Normalize(mean=mean, std=std)(inp)

        mean = [0.6665438079849557]
        std = [0.000683432086931044]
        depth = Normalize(mean=mean, std=std)(depth)

        x = torch.cat((inp, depth), 0)

        img = Image.open(os.path.join(RESULTS, f"{i}-out.png")).convert("L")
        y = (np.asarray(img) > 128)
        #x, y = torch.tensor(x), torch.tensor(y)
        y = torch.tensor(y)

        y = y.long()
        y = F.one_hot(y, num_classes=2).permute(2, 0, 1)

        return x.float(), y.float()


class BrickDatasetAugmented(torch.utils.data.Dataset):
    def __init__(self, root=RESULTS):
        self.root = root

    def __len__(self):
        return BOARD_COUNT * AUGMENTED_COUNT

    def __getitem__(self, i):
        img_id = i // 10
        aug_id = i % 10
        depth = np.load(os.path.join(RESULTS, f"{img_id}.npy"))
        depth = torch.tensor(depth)
        depth = torch.unsqueeze(depth, 0)

        img = Image.open(os.path.join(AUGMENTED, f"{img_id}-{aug_id}-augmented.png"))
        inp = torch.tensor(np.asarray(img)[:, :, :3] * 1.0)
        inp = img_to_net(inp)

        mean = [8.55112426, 187.17984377, 3.40404531]
        std =  [43.36357464, 19.35027745, 17.2615793]
        inp = Normalize(mean=mean, std=std)(inp)

        mean = [0.6665438079849557]
        std = [0.000683432086931044]
        depth = Normalize(mean=mean, std=std)(depth)

        x = torch.cat((inp, depth), 0)

        img = Image.open(os.path.join(RESULTS, f"{img_id}-out.png")).convert("L")
        y = (np.asarray(img) > 128)
        #x, y = torch.tensor(x), torch.tensor(y)
        y = torch.tensor(y)

        y = y.long()
        y = F.one_hot(y, num_classes=2).permute(2, 0, 1)

        return x.float(), y.float()


if __name__ == "__main__":
    #mean: [  8.55112426 187.17984377   3.40404531]
    #std: [43.36357464 19.35027745 17.2615793 ]

    #mean, std = Experiments.mean_std_depth()
    #print(f"mean: {mean}")
    #print(f"std: {std}")

    #mean: 0.6665438079849557
    #std: 0.000683432086931044



    data = BrickDatasetAugmented()
    print(len(data))
    print(data[0])

