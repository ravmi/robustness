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
from model import PickingSegmentationResnet
from loggers import ClearMLLogger

## Parsing ##
task = Task.init(project_name="robustness", task_name="experiment_test", reuse_last_task_id=False)
logger = ClearMLLogger("loss_per_epoch")

parser = argparse.ArgumentParser(description='Choose your hyperparameters.')

parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
parser.add_argument('--batch_size', type=int, default=16, help="number of images in a single batch")
parser.add_argument('--train_split', type=float, default=0.9,
                    help="what percentage of dataset is going to be in the training set")
parser.add_argument('--robust', action="store_true", help="False for default training, True for PGD")
parser.add_argument('--device', type=str, default="cuda", help="device to run on (cuda/cpu")

parameters = vars(parser.parse_args())
parameters = task.connect(parameters)

device = torch.device(parameters['device'])
epochs = parameters['epochs']
lr = parameters['lr']
train_percentage = parameters['train_split']
robust = parameters['robust']
batch_size = parameters['batch_size']

## Prepare dataset ##

data = BrickDataset()
train_size = int(train_percentage * len(data))
val_size = len(data) - train_size

train, val = (Subset(data, range(train_size)), Subset(data, range(train_size, train_size+val_size)))

data_augmented = BrickDatasetAugmented()
# 10 is very specific for the dataset. Every sample in BrickDataset has 10 samples in BrickDataSetAugmented,
# the order is important. We want samples in val and aug to match
data_augmented = Subset(data_augmented, range(train_size * 10, len(data_augmented)))

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
aug_loader = DataLoader(data_augmented, batch_size=batch_size, shuffle=True)

## Prepare the model and the optimizer ##
criterion = nn.BCELoss()
net = PickingSegmentationResnet(criterion, device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.)

metrics = [
    ("pixel_accuracy", Metric("pixel_accuracy")),
    ("precision", Metric("precision")),
    ("recall", Metric("recall")),
    ("balanced", Metric("balanced"))
]

                logger.report_scalar(f"{experiment_name}/loss_per_step", "loss", loss)
            logger.report_scalar(f"{experiment_name}/loss_per_epoch", "loss", np.asarray(losses).mean())


for epoch in range(epochs):
    losses = list()
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
        losses.append(loss.item())
        logger.report_scalar("train/loss_per_step", "loss", loss.item())
        loss.backward()
        optimizer.step()
        #run["train/loss"].log(loss.item())
    logger.report_scalar("train/loss_per_epoch", "loss", np.asarray(losses).mean())

    net.evaluate(val_loader, metrics, log, "val")
    net.evaluate(aug_loader, metrics, log, "aug")
    #evaluate(net, val_loader, device, "val")
    #evaluate(net, aug_loader, device, "aug")
