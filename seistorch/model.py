
import importlib
import traceback
import warnings

import numpy as np
import torch
from yaml import load

from .cell import WaveCell
from .default import ConfigureCheck
from .compile import SeisCompile
from .geom import WaveGeometryFreeForm
from .rnn import WaveRNN
from .utils import set_dtype, update_cfg, to_tensor

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

def build_model(config_path, 
                device = "cuda", 
                mode="forward", 
                source_encoding=False, 
                commands=None, 
                logger=None, 
                backend=None):

    assert mode in ["forward", "inversion", "rtm"], f"No such mode {mode}!"

    # Load the configure file
    with open(config_path, 'r') as ymlfile:
        cfg = load(ymlfile, Loader=Loader)

    # update the configure file
    VEL_PATH = cfg['geom']['initPath'] if mode == 'inversion' else cfg['geom']['truePath']
    cfg.update({'VEL_PATH': VEL_PATH})

    ConfigureCheck(cfg, mode=mode, args=commands)

    # Try to get the data shape by vel model
    try:
        for path in VEL_PATH.values():
            if path is None:
                continue
            d = np.load(path)
            if d.ndim == 2: ny, nx = np.load(path).shape; nz = 0
            if d.ndim == 3: nz, nx, ny = np.load(path).shape    
            if ny is not None and nx is not None:
                cfg['geom'].update({'Nx': nx})
                cfg['geom'].update({'Ny': ny})
                cfg['geom'].update({'Nz': nz})
                break
    except Exception as e:
        traceback.print_exc()
        print(e)

    set_dtype(cfg['dtype'])

    'update_cfg must be called since the width of pml need be added to Nx and Ny'
    cfg = update_cfg(cfg, device=device)
    cfg['task'] = mode

    if cfg['seed'] is not None:
        'Sets the seed for generating random numbers. Returns a torch.Generator object.'
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

    # Set up geometry
    geom  = WaveGeometryFreeForm(mode=mode, logger=logger, **cfg)
    geom.inversion = mode == "inversion"

    # Import cells
    module_name = f"seistorch.equations{geom.ndim}d.{cfg['equation']}"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Cannot found cell '{module_name}'. Please check your equation in configure file.")
        exit()
    # Import the forward and backward functions with the specified equation
    forward_func = getattr(module, "_time_step", None)
    backward_func = getattr(module, "_time_step_backward", None)

    # Compile the forward and backward functions
    compile = SeisCompile(logger=logger)
    forward_func = compile.compile(forward_func)
    backward_func = compile.compile(backward_func)

    forward_func.ACOUSTIC2nd = True if cfg['equation'] == "acoustic" else False
    # Build Cell
    cell = WaveCell(geom, forward_func, backward_func)
    # Build RNN
    model = WaveRNN(cell, source_encoding)

    #return cfg, model
    return cfg, model