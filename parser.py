import argparse

parser = argparse.ArgumentParser(description='Choose your hyperparameters.')

parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
parser.add_argument('--batch_size', type=int, default=16, help="number of images in a single batch")
parser.add_argument('--train_split', type=float, default=0.9, help="what percentage of dataset is going to be in the training set")
parser.add_argument('--robust', action="store_true", help="False for default training, True for PGD")
parser.add_argument('--device', type=str, default="cuda", help="device to run on (cuda/cpu")