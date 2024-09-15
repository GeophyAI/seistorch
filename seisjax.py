"""Perform full waveform inversion."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding"

import jax
from jax.example_libraries.optimizers import adam
import optax

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True

import argparse

import numpy as np
import torch, tqdm
import torch.distributed as dist
from yaml import dump, load
from functools import partial

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import RandomSampler
import seistorch
from seistorch.eqconfigure import Shape
from seistorch.distributed import task_distribution_and_data_reception
from seistorch.io import SeisIO, DataLoader
from torch.utils.tensorboard import SummaryWriter
from seistorch.array import SeisArray
from seistorch.log import SeisLog
from seistorch.coords import single2batch
from seistorch.signal import SeisSignal, generate_arrival_mask
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.utils import (DictAction, to_tensor, nestedlist2np, nestedlist2tensor)
from seistorch.dataset import OBSDataset, NumpyLoader, RandomSampler, DataLoaderJAX
from seistorch.type import TensorList
from seistorch.process import PostProcess
from seistorch.parser import fwi_parser as parser

if __name__ == "__main__":

    # dist.init_process_group("nccl")
    def get_default_device():
        return jax.config.jax_default_device or jax.local_devices()[0]
    
    args = parser().parse_args()

    # delete later
    dev = get_default_device()
    print('Current device:', dev)
   
    # Build model
    cfg, model = build_model(args.config, device=str(dev), mode='inversion', source_encoding=False, commands=args)

    seislog = SeisLog(backend="LOCAL")
    seisio = SeisIO(cfg)
    seissignal = SeisSignal(cfg, seislog)
    setup = SeisSetup(cfg, args, seislog)
    postprocess = PostProcess(model, cfg, args)
    
    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list = seisio.read_geom(cfg)

    # Setup wavelet
    x = setup.setup_wavelet()
    """CONFIGURES"""
    EPOCH_PER_SCALE = cfg['training']['N_epochs']
    ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]
    MINIBATCH = cfg['training']['minibatch']
    MULTISCALES = cfg['geom']['multiscale']
    IMPLICIT = cfg['training']['implicit']['use']
    SCALE_COUNTS = len(MULTISCALES)
    SHOTS_PER_EPOCH = cfg['training']['batch_size'] # USE SHOTS_PER_EPOCH for GRADIENT
    FORDER = cfg['training']['filter_ord']
    # UPDATE THE CONFIGURATION FILE
    cfg['loss'] = args.loss
    cfg['ROOTPATH'] = ROOTPATH
    cfg['training']['lr'] = args.lr
    cfg['training']['optimizer'] = args.opt
    cfg['gradient_cut'] = args.grad_cut
    cfg['gradient_smooth'] = args.grad_smooth

    STEP_PER_EPOCH = args.step_per_epoch
    MULTISCALES = cfg['geom']['multiscale']

    if True:
        os.makedirs(ROOTPATH, exist_ok=True)
        seisio.write_cfg(f"{ROOTPATH}/configure.yml", cfg)
        pbar = tqdm.tqdm(total=EPOCH_PER_SCALE)
        writer = SummaryWriter(os.path.join(ROOTPATH, "logs"))

    """Load obs data"""
    obs0 = OBSDataset(cfg['geom']['obsPath'], 
                      dkey='shot', 
                      srclist=src_list,
                      reclist=rec_list,
                      freqs=None, 
                      PMLN=cfg['geom']['boundary']['width'], 
                      MULTIPLE=cfg['geom']['multiple'])
    nshots = len(obs0)

    SHOTS_PER_GPU = SHOTS_PER_EPOCH#//size

    # obsloader = NumpyLoader(obs0, batch_size=SHOTS_PER_GPU, shuffle=False, sampler=obssampler)
    obsloader = DataLoaderJAX(obs0, batch_size=SHOTS_PER_GPU, shuffle=False, sampler=None)

    # Jax-based optimizer
    # opt_init, opt_update, get_params = adam(10.0)
    # opt_state = opt_init(model.parameters())

    # Optax-based optimizer
    # opt = optax.adam(10.0)
    opt = setup.setup_optimizer_jax()
    opt_state = opt.init(model.parameters())

    rng_key = jax.random.PRNGKey(20240908)
    criterions = setup.setup_criteria()

    params = model.parameters()

    @partial(jax.jit, static_argnums=(4,))
    def step(epoch, opt_state, rng_key, params, freqs):

        rng_key, subkey = jax.random.split(rng_key)

        """Get the data"""
        obsloader.reset_key(subkey)
        _obs, _src, _rec, _shots = next(iter(obsloader))
        _src = _src.T

        bps = SHOTS_PER_GPU//STEP_PER_EPOCH # batchsize_per_step
        
        assert bps > 0, f"Num. of tasks per GPU is {SHOTS_PER_GPU}, but step per epoch is {STEP_PER_EPOCH}."
        loss_all_batch = 0.

        # Reset the parameters
        # model.set_parameters(get_params(opt_state)) # Jax-based updates
        model.set_parameters(params) # Optax-based updates

        for bps_idx in range(STEP_PER_EPOCH):
            start = bps_idx*bps
            end = (bps_idx+1)*bps

            obs = _obs[start:end]
            src = [s[start:end] for s in _src] # [list,]
            rec = _rec[start:end] # (batchsize, ndim, nrecs)
            shots = _shots[start:end]

            batched_source, batched_probes = single2batch(src, rec, cfg, 'cpu') # padding, in batch
            
            obs = SeisArray(obs).filter(cfg['geom']['dt'], freqs, FORDER, axis=1)

            def loss(params):
                syn = model(x, None, batched_source, batched_probes, parameters=params)
                # apply filter
                syn = SeisArray(syn).filter(cfg['geom']['dt'], freqs, FORDER, axis=1)
                return criterions(syn, obs), syn
                
            def compute_gradient(params):
                return jax.value_and_grad(loss, has_aux=True)(params)

            (_loss, syn), gradient = compute_gradient(model.parameters())

            # Jax backend update
            # opt_state = opt_update(epoch, gradient, opt_state)
            
            # Optax backend update
            updates, opt_state = opt.update(gradient, opt_state)
            params = optax.apply_updates(model.parameters(), updates)

            loss_all_batch += _loss

        return loss_all_batch, opt_state, params, rng_key, gradient, syn, obs

    for epoch in range(EPOCH_PER_SCALE*SCALE_COUNTS):

        idx_freq, local_epoch = divmod(epoch, EPOCH_PER_SCALE)

        if local_epoch==0:
            pbar.reset()

        loss, opt_state, params, rng_key, gradient, syn, obs = step(epoch, opt_state, rng_key, params, freqs=MULTISCALES[idx_freq])

        np.save(f"{ROOTPATH}/inverted{epoch:03d}.npy", params)
        np.save(f"{ROOTPATH}/gradient{epoch:03d}.npy", gradient)
        # np.save(f"{ROOTPATH}/syn{epoch:03d}.npy", syn)
        # np.save(f"{ROOTPATH}/obs{epoch:03d}.npy", obs)
        pbar.update(1)
