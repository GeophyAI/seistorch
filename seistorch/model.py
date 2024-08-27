
import importlib

import numpy as np
import torch
from yaml import load

from .cell import WaveCell
from .compile import SeisCompile
from .default import ConfigureCheck
from .eqconfigure import Parameters
from .geom import WaveGeometryFreeForm
from .rnn import WaveRNN
from .utils import set_dtype, update_cfg


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
    try:
        cfg['geom']['source_illumination'] = commands.source_illumination
    except:
        cfg['geom']['source_illumination'] = False
    use_multiple = cfg['geom']['multiple']

    ConfigureCheck(cfg, mode=mode, args=commands)

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
    
    if use_multiple:
        backward_key = '_time_step_backward_multiple'
    else:
        backward_key = '_time_step_backward'
    forward_func = getattr(module, "_time_step", None)
    backward_func = getattr(module, backward_key, None)

    if backward_func is None:
        raise ImportError(f"Cannot found backward function '{module_name}{backward_key}'. Please check your equation in configure file.")

    # Compile the forward and backward functions
    compile = SeisCompile(logger=logger)
    forward_func = compile.compile(forward_func)
    backward_func = compile.compile(backward_func)

    forward_func.ACOUSTIC2nd = True if cfg['equation'] in Parameters.secondorder_equations() else False
    # Build Cell
    cell = WaveCell(geom, forward_func, backward_func)
    # Build RNN
    model = WaveRNN(cell, source_encoding)

    #return cfg, model
    return cfg, model