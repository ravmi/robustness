from torch import nn
import torchvision
import torch
import numpy as np

class PickingSegmentationResnet(nn.Module):
    def __init__(self, criterion, device):
        super(PickingSegmentationResnet, self).__init__()

        resnet = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
        resnet.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = resnet
        self.criterion = criterion
        self.device = device
        self.to(device)

    def forward(self, x):
        predicted = self.resnet.forward(x)['out']
        return nn.Softmax(dim=1)(predicted)

    def train(self):
        self.resnet.train()

    def eval(self):
        self.resnet.eval()

    def evaluate(self, loader, metrics, logger, experiment_name="val"):
        self.eval()
        with torch.no_grad():
            losses = list()
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                predicted = self.forward(x)
                loss = self.criterion(predicted, y).item()
                logger.report_scalar(f"{experiment_name}/loss_per_step", "loss", loss)
                losses.append(loss)

                for name, metric in metrics:
                    metric.measure(
                        predicted.detach().cpu().numpy(),
                        y.detach().cpu().numpy())

            logger.report_scalar(f"{experiment_name}/loss_per_epoch", "loss", np.asarray(losses).mean())

            for m in metrics:
                logger.report_scalar(f"{experiment_name}/acc", m.name, m.total())
