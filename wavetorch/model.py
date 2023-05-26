
import importlib

import numpy as np
import torch
from yaml import load

from .cell import WaveCell
from .geom import WaveGeometryFreeForm
from .rnn import WaveRNN
from .utils import set_dtype, update_cfg

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

def build_model(config_path, device = "cuda", mode="forward"):

    assert mode in ["forward", "inversion"], f"No such mode {mode}!"

    # Load the configure file
    with open(config_path, 'r') as ymlfile:
        cfg = load(ymlfile, Loader=Loader)

    VEL_PATH = cfg['geom']['initPath'] if mode == 'inversion' else cfg['geom']['truePath']
    cfg.update({'VEL_PATH': VEL_PATH})

    # Try to get the data shape by vel model
    try:
        ny, nx = np.load(VEL_PATH['vp']).shape
        cfg['geom'].update({'Nx': nx})
        cfg['geom'].update({'Ny': ny})
    except Exception as e:
        print(e)

    set_dtype(cfg['dtype'])

    'update_cfg must be called since the width of pml need be added to Nx and Ny'
    cfg = update_cfg(cfg, device=device)

    if cfg['seed'] is not None:
        'Sets the seed for generating random numbers. Returns a torch.Generator object.'
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

    # Set up geometry
    geom  = WaveGeometryFreeForm(**cfg)
    geom.inversion = mode == "inversion"

    # Import cells
    module_name = f"wavetorch.equations.{cfg['equation']}"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Cannot found cell '{module_name}'. Please check your equation in configure file.")
        exit()
    # Import the forward and backward functions with the specified equation
    forward_func = getattr(module, "_time_step", None)
    backward_func = getattr(module, "_time_step_backward", None)
    # Build Cell
    cell = WaveCell(geom, forward_func, backward_func)
    # Build RNN
    model = WaveRNN(cell)

    return cfg, model