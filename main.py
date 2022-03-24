
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


from pgd import projected_gradient_descent
import torchvision
import torch
from torch import nn
from dataloader import BrickDataset, BrickDatasetAugmented
from utils import img_to_net
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import neptune.new as neptune
run = neptune.init(
    project="rm360179/robustness",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==",
)


device = torch.device("cuda:0")
#device = torch.device("cpu")
epochs = 20
epochs = 30
#lr = 5e-5
lr = 1e-5
lr = 1e-4
train_percentage = 0.9
robust = True
robust = False
robust = True
batch_size = 16

run['lr'] = lr
run['epochs'] = epochs
run['train_percentage'] = train_percentage
run['robust'] = robust
run['batch_size'] = batch_size





data = BrickDataset()
train_size = int(train_percentage * len(data))
val_size = len(data) - train_size
print(len(data))
#test_size = (len(data) - train_size) // 2
#val_size = len(data) - train_size - test_size

#train, val, test = torch.utils.data.random_split(data, [train_size, val_size, test_size])
train, val = (Subset(data, range(train_size)), Subset(data, range(train_size, train_size+val_size)))

data_augmented = BrickDatasetAugmented()
data_augmented = Subset(data_augmented, range(train_size * 10, len(data_augmented)))

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
aug_loader = DataLoader(data_augmented, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test, batch_size=32, shuffle=True)

resnet = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
resnet.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.to(device)

criterion = nn.BCELoss()




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet

    def forward(self, x):
        predicted = resnet.forward(x)['out']
        return torch.nn.Softmax(dim=1)(predicted)

    def train(self):
        self.resnet.train()
    def eval(self):
        self.resnet.eval()


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.)


def evaluate(net, loader, device, logname):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            #predicted = net.forward(x)['out']
            #predicted = torch.nn.Softmax(dim=1)(predicted)
            predicted = net.forward(x)

            argp = torch.argmax(predicted, axis=1).reshape(-1)
            argy = torch.argmax(y, axis=1).reshape(-1)
            correct += torch.sum(argp == argy).item()
            total += len(argy)

            loss = criterion(predicted, y)
            run[f"{logname}/loss"].log(loss.item())
        run[f"{logname}/acc"].log(correct/total)


for epoch in range(epochs):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

       
        optimizer.zero_grad()

        if robust:
            net.eval()
            x = projected_gradient_descent(net, x, y, criterion,
                    num_steps=40, step_size=0.01,
                    eps=0.3, eps_norm='inf',
                    step_norm='inf')
            net.train()
        

        optimizer.zero_grad()

        predicted = net.forward(x)


        loss = criterion(predicted, y)
        loss.backward()
        optimizer.step()
        run["train/loss"].log(loss.item())

    evaluate(net, val_loader, device, "val")
    evaluate(net, aug_loader, device, "aug")

run.stop()
