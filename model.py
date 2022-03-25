from torch import nn
import torchvision
import torch

class PickingSegmentationResnet(nn.Module):
    def __init__(self, criterion, device):
        super(PickingSegmentationResnet, self).__init__()

        resnet = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
        resnet.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = resnet
        self.criterion = criterion
        self.device = device

    def forward(self, x):
        predicted = self.resnet.forward(x)['out']
        return nn.Softmax(dim=1)(predicted)

    def train(self):
        self.resnet.train()

    def eval(self):
        self.resnet.eval()

    def evaluate(self, loader, metrics, logger):
        self.eval()
        with torch.no_grad():
            for x, y in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                predicted = self.forward(x)
                loss = self.criterion(predicted, y)

                for name, metric in metrics:
                    metric.measure(
                        predicted.detach().cpu().numpy(),
                        y.detach.cpu().numpy())

                #logger.report_scalar()
                #run[f"{logname}/loss"].log(loss.item())

            for name, m in metrics:
                logger.report_scalar(name, m.total())
                #run[f"{logname}/{m.metric_name}"].log(m.total())
