
import importlib

import numpy as np
import torch
from yaml import load

from .cell import WaveCellJax, WaveCellTorch
from .default import ConfigureCheck
from .eqconfigure import Parameters
from .geom import WaveGeometryFreeForm
from .rnn import WaveRNN, WaveRNNJAX
from .utils import set_dtype
from .setup import setup_we_equations, setup_domain_shape


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

    if 'spatial_order' not in cfg.get('geom', {}):
        cfg['geom']['spatial_order'] = 2

    # Set backend A.D. framework
    backend = cfg.get('backend', 'torch')
    cfg.setdefault('backend', backend)

    use_jax = (backend == 'jax')
    use_torch = (backend == 'torch')
    use_multiple = cfg['geom']['multiple']

    ConfigureCheck(cfg, mode=mode, args=commands)

    set_dtype(cfg['dtype'])

    # update_cfg must be called since the width of pml need be added to Nx and Ny
    cfg = setup_domain_shape(cfg)
    cfg.setdefault('device', device)
    cfg['task'] = mode

    if cfg['seed'] is not None:
        'Sets the seed for generating random numbers. Returns a torch.Generator object.'
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

    # Set up geometry
    geom = WaveGeometryFreeForm(mode=mode, logger=logger, **cfg)
    geom.inversion = mode == "inversion"

    forward_func, backward_func = setup_we_equations(use_jax, use_torch, use_multiple, geom.ndim, cfg['equation'], logger)

    forward_func.ACOUSTIC2nd = True if cfg['equation'] in Parameters.secondorder_equations() else False

    # Build Cell
    module = importlib.import_module("seistorch.cell")
    cell = getattr(module, f"WaveCell{backend.capitalize()}")(geom, forward_func, backward_func)

    if use_torch:
        model = WaveRNN(cell, source_encoding)
    if use_jax:
        model = WaveRNNJAX(cell, source_encoding)

    return cfg, model