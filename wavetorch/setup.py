import os

import numpy as np
import torch

from wavetorch.eqconfigure import Parameters
from wavetorch.loss import Loss
from wavetorch.utils import read_pkl, ricker_wave, to_tensor


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
    optimizers = torch.optim.Adam(paras_for_optim, betas=(0.9, 0.999), eps=1e-22)
    # Setup the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, epoch_decay, last_epoch=- 1, verbose=False)

    return optimizers, lr_scheduler

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

def setup_wavelet(cfg):
    """Setup the wavelet for the simulation.

    Args:
        cfg (_type_): The configuration file.

    Returns:
        torch.Tensor: Tensor containing the wavelet.
    """
    if not cfg["geom"]["wavelet"]:
        print("Using wavelet func.")
        x = ricker_wave(cfg['geom']['fm'], cfg['geom']['dt'], cfg['geom']['nt'])
    else:
        print("Loading wavelet from file")
        x = to_tensor(np.load(cfg["geom"]["wavelet"]))
    # Save the wavelet
    np.save(os.path.join(cfg['ROOTPATH'], "wavelet.npy"), x.cpu().numpy())
    return x