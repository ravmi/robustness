from robustness.datasets import DATASETS
from robustness.model_utils import make_and_restore_model
from robustness.train import train_model
from robustness.defaults import check_and_fill_args
from robustness.tools import constants, helpers
from robustness import defaults
from robustness.datasets import CIFAR

from cox import utils
from cox import store
print('asdf')
import torch as ch
from argparse import ArgumentParser
import os

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .train import train_model, eval_model
    from .tools import constants, helpers
    from . import defaults, __version__
    from .defaults import check_and_fill_args
except:
    raise ValueError("Make sure to run with python -m (see README.md)")

parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)




'''
parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
# Note that we can add whatever extra arguments we want to the parser here
args = parser.parse_args()


#args = check_and_fill_args(args, defaults.TRAINING_ARGS, DATASETS['imagenet'])
args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, CIFAR)
print(args)
print(DATASETS)


print(args)
print('ee')
'''
