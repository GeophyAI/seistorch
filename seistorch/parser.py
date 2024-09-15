import argparse
from seistorch.utils import DictAction

def coding_fwi_parser():
    """Construct the parser for coding FWI

    Returns:
        parser: the parser for coding FWI
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, 
                        help='Configuration file for geometry, training, and data preparation')
    parser.add_argument('--num_threads', type=int, default=2,
                        help='Number of threads to use')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA to perform computations')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='which gpu is used for calculation')
    parser.add_argument('--checkpoint', type=str,
                        help='checkpoint path for resuming training')
    parser.add_argument('--opt', choices=['adam', 'lbfgs', 'cg', 'steepestdescent'], default='adam',
                        help='optimizer (adam)')
    parser.add_argument('--loss', action=DictAction, nargs="+",
                        help='loss dictionary')
    parser.add_argument('--save-path', default='',
                        help='the root path for saving results')
    parser.add_argument('--lr', action=DictAction, nargs="+",
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=-1,
                        help='batch size for coding')
    parser.add_argument('--grad-smooth', action='store_true',
                        help='Smooth the gradient or not')
    parser.add_argument('--grad-cut', action='store_true',
                        help='Cut the boundaries of gradient or not')
    parser.add_argument('--disable-grad-clamp', action='store_true',
                        help='Clamp the gradient using quantile or not')
    parser.add_argument('--mode', choices=['inversion'], default='inversion',
                        help='forward modeling, inversion or reverse time migration mode')
    parser.add_argument('--source-encoding', action='store_true', default=True,
                        help='PLEASE DO NOT CHANGE THE DEFAULT VALUE.')
    parser.add_argument('--filteratlast', action='store_true', 
                        help='Filter the wavelet at the last step or not')

    return parser

def fwi_parser():
    """Construct the parser for FWI

    Returns:
        parser: the parser for FWI
    """
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, 
                        help='Configuration file for geometry, training, and data preparation')
    parser.add_argument('--num_threads', type=int, default=2,
                        help='Number of threads to use')
    parser.add_argument('--num-batches', type=int, default=1,
                        help='Number of batches to use')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA to perform computations')
    parser.add_argument('--opt', choices=['adam', 'lbfgs', 'steepestdescent', 'cg'], default='adam',
                        help='optimizer (adam)')
    parser.add_argument('--save-path', default='',
                        help='the root path for saving results')
    parser.add_argument('--loss', action=DictAction, nargs="+",
                        help='loss dictionary')
    parser.add_argument('--lr', action=DictAction, nargs="+",
                        help='learning rate')
    parser.add_argument('--mode', choices=['forward', 'inversion', 'rtm'], default='forward',
                        help='forward modeling, inversion or reverse time migration mode')
    parser.add_argument('--modelparallel', action='store_true',
                        help='Split the model to various GPUs')
    parser.add_argument('--grad-cut', action='store_true',
                        help='Cut the boundaries of gradient or not')
    parser.add_argument('--grad-smooth', action='store_true',
                        help='Smooth the gradient or not')
    parser.add_argument('--grad-clip', action='store_true', default=True,
                        help='Clip the gradient or not')
    parser.add_argument('--source-illumination', action='store_true',
                        help='Use source illumination or not')
    parser.add_argument('--filteratfirst', action='store_true', 
                        help='Filter the wavelet at the first step or not')
    parser.add_argument('--obsnofilter', action='store_true', 
                        help='Do not filter the observed data')
    parser.add_argument('--clipvalue', type=float, default=0.02)
    parser.add_argument('--step-per-epoch', type=int, default=1)
    parser.add_argument('--source-encoding', action='store_true', default=False,
                        help='PLEASE DO NOT CHANGE THE DEFAULT VALUE.')

    return parser