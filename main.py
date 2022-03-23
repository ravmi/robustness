
'''
normalization:
        ps_locator.heightmap_transforms = ImageTransformationComposite([
        DepthInversion(),
        DepthNormalization(mean=0.001250115736038424, std=0.007071534325436761),
        DepthChannelTiling(),
        RGBNormalization(means=[110.38053718566894, 107.84343673706054, 109.98833946228028],
                         stds=[68.15320649910386, 69.23072564755397, 70.3886151478994]),
    ])
'''

# normalization



import torchvision
import torch
from torch import nn
from dataloader import BrickDataset
from utils import img_to_net
import torch.nn.functional as F
from torch.utils.data import DataLoader


device = torch.device("cuda:0")
device = torch.device("cpu")

training_data = BrickDataset(device)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

net = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
net.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.to(device)

net.train()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.)

epochs = 0
criterion = nn.BCELoss()

for epoch in range(epochs):
    for x, y in train_dataloader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        print(x.type())
        print(y.type())

        predicted = net.forward(x)['out']
        predicted = torch.nn.Softmax(dim=1)(predicted)

        loss = criterion(predicted, y)
        loss.backward()
        optimizer.step()
        print(loss)
    print(loss)
