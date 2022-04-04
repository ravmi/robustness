import torch
import numpy as np


x = np.arange(10)
y = x * 1.0
x = np.random.randn(10, 2, 2)

print(torch.is_tensor(x))