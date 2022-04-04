import torch
import numpy as np
from dataloader import BrickDataset


data = BrickDataset()
print(len(data))
print(data[0][0])
print(data[0][1])
