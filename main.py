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
from clearml import Task, Logger
import argparse
from metrics import Metric
import numpy as np

task = Task.init(project_name="robustness", task_name="experiment_test")
log = Logger.current_logger()

parser = argparse.ArgumentParser(description='Choose your hyperparameters.')

parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
parser.add_argument('--batch_size', type=int, default=16, help="number of images in a single batch")
parser.add_argument('--train_split', type=float, default=0.9, help="what percentage of dataset is going to be in the training set")
parser.add_argument('--robust', action="store_true", help="False for default training, True for PGD")
parser.add_argument('--device', type=str, default="cuda", help="device to run on (cuda/cpu")


'''
run = neptune.init(
    project="rm360179/robustness",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==",
)
'''




parameters = vars(parser.parse_args())
parameters = task.connect(parameters)


print(args)
print(vars(args))


device = torch.device("cuda:0")
epochs = 20
epochs = 100
#lr = 5e-5
lr = 1e-5
lr = 1e-4
train_percentage = 0.9
robust = False
robust = False
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
        pa = Metric("pixel_accuracy")
        pr = Metric("precision")
        re = Metric("recall")
        ba = Metric("balanced")
        metrics = [pa, pr, re, ba]

        net.eval()
        #correct = 0
        #total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            #predicted = net.forward(x)['out']
            #predicted = torch.nn.Softmax(dim=1)(predicted)
            predicted = net.forward(x)

            #argp = torch.argmax(predicted, axis=1).reshape(-1)
            #argy = torch.argmax(y, axis=1).reshape(-1)
            for m in metrics:
                m.measure(predicted.detach().cpu().numpy(), y.detach().cpu().numpy())
            #correct += torch.sum(argp == argy).item()
            #total += len(argy)

            loss = criterion(predicted, y)
            run[f"{logname}/loss"].log(loss.item())

        for m in metrics:
            run[f"{logname}/{m.metric_name}"].log(m.total())


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
