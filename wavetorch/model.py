
import torch
import numpy as np
from yaml import load
from .utils import set_dtype, update_cfg
from .cell_elastic import WaveCell as WaveCellElastic
from .cell_acoustic import WaveCell as WaveCellAcoustic
from .cell_viscoacoustic import WaveCell as WaveCellViscoacoustic

from .rnn import WaveRNN

from .geom import WaveGeometryFreeForm
from .setup_source_probe import setup_src_coords_customer, setup_probe_coords_customer, get_sources_coordinate_list
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

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

    # Set up probes
    probes = setup_probe_coords_customer(cfg)

    # Set up geometry
    geom  = WaveGeometryFreeForm(**cfg)

    # Add the key 'equation' to the configure file
    cfg['equation'] = geom.equation

    # Branch

    assert cfg['equation'] in ['acoustic', 'elastic', 'viscoacoustic'], f"Cannot find such equation type {cfg['equation']}"

    WaveCell = {"elastic": WaveCellElastic, 
                "acoustic": WaveCellAcoustic,
                "viscoacoustic": WaveCellViscoacoustic}
    
    cell = WaveCell[cfg['equation']](geom)

    model = WaveRNN(cell, probes=probes)

    return cfg, model