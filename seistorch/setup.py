import os

import numpy as np
import torch

from seistorch.eqconfigure import Parameters
from seistorch.loss import Loss
from seistorch.utils import read_pkl, ricker_wave, to_tensor
from seistorch.source import WaveSource
from seistorch.probe import WaveIntensityProbe

def setup_acquisition(shots, src_list, rec_list, cfg, *args, **kwargs):

    sources, receivers = [], []

    for shot in shots:
        src = setup_src_coords(src_list[shot], cfg['geom']['pml']['N'], cfg['geom']['multiple'])
        rec = setup_rec_coords(rec_list[shot], cfg['geom']['pml']['N'], cfg['geom']['multiple'])
        sources.append(src)
        receivers.extend(rec)

    return sources, receivers

def setup_criteria(cfg: dict, loss: dict, *args, **kwargs):
    """Setup the loss functions for the model

    Args:
        cfg (dict): The configuration file.
        loss (dict): The losses specified in the running arguments.
    Returns:
        torch.nn.module: The specified loss function.
    """
    ACOUSTIC = cfg['equation'] == 'acoustic'
    # The parameters needed to be inverted
    loss_names = set(loss.values())
    MULTI_LOSS = len(loss_names) > 1
    if not MULTI_LOSS or ACOUSTIC:
        print("Only one loss function is used.")
        criterions = Loss(list(loss_names)[0]).loss(cfg)
    else:
        criterions = {k:Loss(v).loss(cfg) for k,v in loss.items()}
        print(f"Multiple loss functions are used:\n {criterions}")
    return criterions

def setup_optimizer(model, cfg, idx_freq=0, implicit=False, *args, **kwargs):
    """Setup the optimizer for the model

    Args:
        model (RNN): The model to be optimized.
        cfg (dict): The configuration file.
    """
    lr = cfg['training']['lr']
    epoch_decay = cfg['training']['lr_decay']
    scale_decay = cfg['training']['scale_decay']
    pars_need_by_eq = Parameters.valid_model_paras()[cfg['equation']]
    pars_need_invert = [k for k, v in cfg['geom']['invlist'].items() if v]

    # Setup the learning rate for each parameter
    paras_for_optim = []
    # 
    if not implicit:
        for para in pars_need_by_eq:
            # Set the learning rate for each parameter
            _lr = 0. if para not in pars_need_invert else lr[para]*scale_decay**idx_freq
            paras_for_optim.append({'params': model.cell.get_parameters(para), 
                                    'lr':_lr})
    if implicit:
        _lr = 1e-4
        paras_for_optim.append({'params': model.cell.geom.siren.parameters(), 
                                'lr':_lr})

    # Setup the optimizer
    if 'fatt' in cfg['loss'].values():
        print("Using first arrival loss")
        for idx in range(len(paras_for_optim)):
            paras_for_optim[idx]['lr'] = 5e2 # For SGD
        optimizers = torch.optim.SGD(paras_for_optim, momentum=0.9)
    else:
        optimizers = torch.optim.Adam(paras_for_optim, betas=(0.9, 0.999), eps=1e-22)

    # Setup the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, epoch_decay, last_epoch=- 1, verbose=False)

    return optimizers, lr_scheduler

def setup_rec_coords(coords, Npml, multiple=False):
    """Setup receiver coordinates.

    Args:
        coords (list): A list of coordinates.
        Npml (int): The number of PML layers.
        multiple (bool, optional): Whether use top PML or not. Defaults to False.

    Returns:
        WaveProbe: A torch.nn.Module receiver object.
    """

    # Coordinate are specified
    keys = ['x', 'y', 'z']
    kwargs = dict()

    # Without multiple
    for key, value in zip(keys, coords):
        kwargs[key] = [v+Npml for v in value]

    # 2D case with multiple
    if 'z' not in kwargs.keys() and multiple:
        kwargs['y'] = [v-Npml for v in kwargs['y']]
    # 3D case with multiple
    if 'z' in kwargs.keys() and multiple:
        raise NotImplementedError("Multiples in 3D case is not implemented yet.")
        #kwargs['z'] = [v-Npml for v in kwargs['z']]

    return [WaveIntensityProbe(**kwargs)]

def setup_src_rec(cfg: dict):
    """Read the source and receiver locations from the configuration file.

    Args:
        cfg (dict): The configuration file.

    Returns:
        tuple: Tuple containing: (source locations, 
        receiver locations of each shot, 
        full receiver locations,
        whether the receiver locations are fixed)
    """
    # Read the source and receiver locations from the configuration file
    assert os.path.exists(cfg["geom"]["sources"]), "Cannot found source file."
    assert os.path.exists(cfg["geom"]["receivers"]), "Cannot found receiver file."
    src_list = read_pkl(cfg["geom"]["sources"])
    rec_list = read_pkl(cfg["geom"]["receivers"])
    assert len(src_list)==len(rec_list), \
        "The lenght of sources and recev_locs must be equal."
    # Check whether the receiver locations are fixed
    fixed_receivers = all(rec_list[i]==rec_list[i+1] for i in range(len(rec_list)-1))
    # If the receiver locations are not fixed, use the model grids as the full receiver locations
    if not fixed_receivers: 
        print(f"Inconsistent receiver location detected.")
        receiver_counts = cfg['geom']['_oriNx']
        rec_depth = rec_list[0][1][0]
        full_rec_list = [[i for i in range(receiver_counts)], [rec_depth]*receiver_counts]
        # TODO: Add a warning here
        # The full receiver list should be the available receivers in rec_list.
    else:
        print(f"Receiver locations are fixed.")
        full_rec_list = rec_list[0]

    return src_list, rec_list, full_rec_list, fixed_receivers

def setup_src_coords(coords, Npml, multiple=False):
    """Setup source coordinates.

    Args:
        coords (list): A list of coordinates.
        Npml (int): The number of PML layers.
        multiple (bool, optional): Whether use top PML or not. Defaults to False.

    Returns:
        WaveSource: A torch.nn.Module source object.
    """
    # Coordinate are specified
    keys = ['x', 'y', 'z']
    kwargs = dict()
    for key, value in zip(keys, coords):
        kwargs[key] = value+Npml
    
    # 2D case with multiple
    if 'z' not in kwargs.keys() and multiple:
        kwargs['y'] -= Npml
    # 3D case with multiple
    if 'z' in kwargs.keys() and multiple:
        raise NotImplementedError("Multiples in 3D case is not implemented yet.")
        # kwargs['z'] -= Npml

    return WaveSource(**kwargs)

def setup_wavelet(cfg):
    """Setup the wavelet for the simulation.

    Args:
        cfg (_type_): The configuration file.

    Returns:
        torch.Tensor: Tensor containing the wavelet.
    """
    if not cfg["geom"]["wavelet"]:
        print("Using wavelet func.")
        x = ricker_wave(cfg['geom']['fm'], 
                        cfg['geom']['dt'], 
                        cfg['geom']['nt'], 
                        cfg['geom']['wavelet_delay'], 
                        inverse=cfg['geom']['wavelet_inverse'])
    else:
        print("Loading wavelet from file")
        x = to_tensor(np.load(cfg["geom"]["wavelet"]))
    # Save the wavelet
    np.save(os.path.join(cfg['ROOTPATH'], "wavelet.npy"), x.cpu().numpy())
    return x