import argparse
import time

from wavetorch.utils import DictAction

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('--num_threads', type=int, default=2,
                    help='Number of threads to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--encoding', action='store_true',
                    help='Use phase encoding to accelerate performance')
parser.add_argument('--gpuid', type=int, default=0,
                    help='which gpu is used for calculation')
parser.add_argument('--checkpoint', type=str,
                    help='checkpoint path for resuming training')
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
parser.add_argument('--opt', choices=['adam', 'lbfgs', 'ncg'], default='adam',
                    help='optimizer (adam)')
parser.add_argument('--loss', action=DictAction, nargs="+",
                    help='loss dictionary')
parser.add_argument('--save-path', default='',
                    help='the root path for saving results')
parser.add_argument('--lr', action=DictAction, nargs="+",
                    help='learning rate')
parser.add_argument('--global-lr', type=int, default=-1,
                    help='learning rate')
parser.add_argument('--batchsize', type=int, default=-1,
                    help='learning rate')
parser.add_argument('--grad-smooth', action='store_true',
                    help='Smooth the gradient or not')
parser.add_argument('--grad-cut', action='store_true',
                    help='Cut the boundaries of gradient or not')
parser.add_argument('--mode', choices=['inversion'], default='inversion',
                    help='forward modeling, inversion or reverse time migration mode')