from torch import nn
import torchvision
import torch
import numpy as np
import config
from utils import net_to_img, unnormalize

class PickingSegmentationResnet(nn.Module):
    def __init__(self, criterion, device):
        super(PickingSegmentationResnet, self).__init__()

        resnet = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
        resnet.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if config.debug:
            resnet.backbone.layer1 = nn.Conv2d(64, 256, kernel_size=(1, 1))
            resnet.backbone.layer2 = nn.Conv2d(256, 512, kernel_size=(1, 1))
            resnet.backbone.layer3 = nn.Conv2d(512, 1024, kernel_size=(1, 1))
            resnet.backbone.layer4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1))

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
                #logger.report_scalar(f"{experiment_name}/loss_per_step", "loss", loss)
                losses.append(loss)

                for metric in metrics:
                    metric.measure(
                        predicted.detach().cpu().numpy(),
                        y.detach().cpu().numpy())

            logger.report_scalar(f"{experiment_name}/loss_per_epoch", "loss", np.asarray(losses).mean())
            for m in metrics:
                logger.report_scalar(f"{experiment_name}/acc", m.metric_name, m.total())
            
            xb = x
            yb = y
            pb = predicted
            for i in range(len(xb)):
                x = xb[i]
                x = x[:3, :, :].detach().cpu()
                x = unnormalize(x)
                y = yb[i].detach().cpu()
                predicted = pb[i].detach().cpu()
                x = net_to_img(x).numpy()
                y = np.argmax(y.numpy(), axis=0)
                predicted = np.argmax(predicted.numpy(), axis=0)
                predicted_as_true = np.sum((predicted == 1))
                predicted_as_false = np.sum((predicted == 0))

                shouldbe_true = np.sum(y == 1)
                shouldbe_false = np.sum(y == 0)

                logger.report_image(f"{experiment_name}/img", f"img{i}", x)
                logger.report_image(f"{experiment_name}/truth", f"img{i}", y * 255)
                logger.report_image(f"{experiment_name}/guessed", f"img{i}", predicted * 255)

                logger.report_scalar(f"{experiment_name}/as_true{i}", f"predicted", predicted_as_true)
                logger.report_scalar(f"{experiment_name}/as_true{i}", f"shouldbe", shouldbe_true)

                logger.report_scalar(f"{experiment_name}/as_false{i}", f"predicted", predicted_as_false)
                logger.report_scalar(f"{experiment_name}/as_false{i}", f"shouldbe", shouldbe_false)


