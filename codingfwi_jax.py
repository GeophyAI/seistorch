"""Perform full waveform inversion."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".90"

import jax
import jax.numpy as jnp

import argparse
import logging
import time
import setproctitle
import torch
import tqdm

import numpy as np
import setproctitle
from torch.utils.tensorboard import SummaryWriter
from yaml import dump
from functools import partial

from seistorch.array import SeisArray
from seistorch.coords import single2batch, offset_with_boundary

from seistorch.eqconfigure import Shape
# from tensorflow.keras.models import load_model
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.log import SeisLog
from seistorch.io import SeisIO
from seistorch.signal import SeisSignal
from seistorch.utils import (DictAction, dict2table, roll_jax, to_tensor)
from seistorch.process import PostProcess
from seistorch.dataset import OBSDataset, NumpyLoader, DataLoaderJAX
from seistorch.parser import coding_fwi_parser as parser

if __name__ == '__main__':

    # dist.init_process_group("nccl")
    def get_default_device():
        return jax.config.jax_default_device or jax.local_devices()[0]
    
    # delete later
    dev = get_default_device()
    print('Current device:', dev)

    args = parser().parse_args()

    args.dev = dev

    seislog = SeisLog("Seisjax", backend="LOCAL")

    'Sets the number of threads used for intraop parallelism on CPU.'

    # Build model
    cfg, model = build_model(args.config, device=str(dev), mode=args.mode, source_encoding=args.source_encoding, commands=args, logger=seislog)
    # model = torch.compile(model)
    seisio = SeisIO(cfg)
    setup = SeisSetup(cfg, args, seislog)
    # Set random seed
    setup.setup_seed()

    # Set the name of the process
    setproctitle.setproctitle("coding_fwi")

    """Short cuts of the configures"""
    DT = cfg['geom']['dt']
    FORDER = cfg['training']['filter_ord']
    EPOCHS = cfg['training']['N_epochs']
    IMPLICIT = cfg['training']['implicit']['use']
    MINIBATCH = cfg['training']['minibatch']
    MULTISCALES = cfg['geom']['multiscale']
    EPOCH_PER_SCALE = cfg['training']['N_epochs']
    SCALE_COUNTS = len(MULTISCALES)
    BATCHSIZE = cfg['training']['batch_size'] if args.batchsize < 0 else args.batchsize
    PARS_NEED_INVERT = [k for k, v in cfg['geom']['invlist'].items() if v]
    ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]

    # Check the working folder
    if not os.path.exists(ROOTPATH):
        os.makedirs(ROOTPATH, exist_ok=True)
    # Configure the logger
    logging.basicConfig(level=logging.DEBUG,  # Set the log level to DEBUG (the lowest level)
                        format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log format
                        filename=f'{ROOTPATH}/log.log',  # Specify the log file name
                        filemode='w')  # Set the file mode to write mode
    writer = SummaryWriter(os.path.join(ROOTPATH, "logs"))
    seislog.print(f"The results will be saving at '{ROOTPATH}'")
    seislog.print(f"BATCHSIZE: {args.batchsize}")
    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list = seisio.read_geom(cfg)
    recs_are_fixed, full_rec_list = setup.setup_fixed_receivers(rec_list)

    postprocess = PostProcess(model, cfg, args)

    cfg["loss"] = args.loss
    cfg["ROOTPATH"] = ROOTPATH
    cfg['training']['lr'] = args.lr
    cfg['training']['batchsize'] = BATCHSIZE
    cfg['training']['optimizer'] = args.opt

    """Write configure file to the inversion folder"""
    seisio.write_cfg(os.path.join(ROOTPATH, "configure.yml"), cfg)

    """# Read the wavelet"""
    x = setup.setup_wavelet()

    seissignal = SeisSignal(cfg, seislog)
    shape = Shape(cfg)

    """Set receivers"""
    # In coding fwi, the probes are set only once, 
    # because they are fixed with respect to moving source.

    temp, probes = offset_with_boundary(jnp.array(src_list), jnp.array(rec_list), cfg)
    _, probes = single2batch(temp[0:1], probes[0:1], cfg, 'cpu') # padding, in batch
    probes.bidx = 0


    """---------------------------------------------"""
    """-------------------INVERSION-----------------"""
    """---------------------------------------------"""

    """Load obs data"""
    obs0 = OBSDataset(cfg['geom']['obsPath'], 
                      dkey='shot', 
                      srclist=src_list,
                      reclist=rec_list,
                      freqs=None, 
                      PMLN=cfg['geom']['boundary']['width'], 
                      MULTIPLE=cfg['geom']['multiple'])
    nshots = len(obs0)
    obsloader = DataLoaderJAX(obs0, batch_size=BATCHSIZE, shuffle=False, sampler=None)
    
    pbar = tqdm.tqdm(total=EPOCH_PER_SCALE)

    """Define the misfit function"""
    # The parameters needed to be inverted
    criterions = setup.setup_criteria()

    # initial paras
    params = model.parameters()

    rng_key = jax.random.PRNGKey(cfg['seed'])

    @partial(jax.jit, static_argnums=(4,))
    def step(epoch, opt_state, rng_key, params, freqs):

        """Reset params"""
        model.set_parameters(params)

        # Allocate the memory for coding data
        coding_obs = jnp.zeros((shape.nt, len(full_rec_list[0]), shape.channels), dtype=jnp.float32)
        coding_wav = jnp.zeros((BATCHSIZE, shape.nt), dtype=jnp.float32)

        keys = jax.random.split(rng_key, BATCHSIZE+1)

        """Get the data"""
        obsloader.reset_key(keys[-1])

        _obs, _src, _rec, _shots = next(iter(obsloader))

        """Offset the source and receiver"""
        _src, _rec = offset_with_boundary(_src, _rec, cfg)
        batched_source, _ = single2batch(_src, _rec, cfg, 'cpu') # padding, in batch

        """Filter the observed data"""
        _obs = SeisArray(_obs).filter(DT, freqs, FORDER, 1)

        """Filter the wavelet"""
        fx = SeisArray(x).filter(DT, freqs, FORDER, 0)

        """Clear the coding data"""
        coding_obs = coding_obs.at[:].set(0.)
        coding_wav = coding_wav.at[:].set(0.)

        """Coding the observed data and wavelet"""
        for shot in range(BATCHSIZE):
            wave_temp, d_temp = roll_jax(fx, _obs[shot], key=keys[shot])
            coding_obs = coding_obs.at[:].add(d_temp)
            coding_wav = coding_wav.at[shot].set(wave_temp)

        def loss(params):
            coding_syn = model(coding_wav, None, super_source=batched_source, super_probes=probes, parameters=params)
            return criterions(coding_syn, coding_obs), coding_syn
        
        def compute_gradient(params):
            return jax.value_and_grad(loss, has_aux=True)(params)

        (_loss, coding_syn), gradient = compute_gradient(params)
        updates, opt_state = opt.update(gradient, opt_state)
        params = optax.apply_updates(params, updates)

        return coding_obs, coding_syn, _shots, keys[0], params, gradient, opt_state, _loss

    """Loop over the epochs"""
    for epoch in range(EPOCH_PER_SCALE*SCALE_COUNTS):

        idx_freq, local_epoch = divmod(epoch, EPOCH_PER_SCALE)
        pbar.set_description(f"F{idx_freq}E{local_epoch}")

        if local_epoch==0:
            """Reset the optimizer at every scale"""
            opt = setup.setup_optimizer_jax(idx_freq=idx_freq)
            opt_state = opt.init(model.parameters())
            pbar.reset()

        outputs = step(epoch, opt_state, rng_key, params, freqs=MULTISCALES[idx_freq])
        obs, syn, _shots, rng_key, params, gradient, opt_state, coding_loss = outputs

        # Get learning rate        
        # filtering = lambda path, value: isinstance(value, jnp.ndarray)
        # learning_rate = optax.tree_utils.tree_get( opt_state, 'learning_rate', filtering=filtering)

        np.save(f"{ROOTPATH}/model_F{idx_freq:02d}E{local_epoch:02d}.npy", params)
        np.save(f"{ROOTPATH}/gradient_F{idx_freq:02d}E{local_epoch:02d}.npy", gradient)
        pbar.set_description(f"F{idx_freq}E{local_epoch}")
        np.save(f"{ROOTPATH}/obs_{epoch}.npy", obs)
        np.save(f"{ROOTPATH}/syn_{epoch}.npy", syn)

        writer.add_scalar("Loss", coding_loss.item(), epoch)
        # writer.add_scalar("Learning Rate", learning_rate.item(), epoch)
        writer.add_histogram('Sample Indices', np.array(_shots), epoch)

        pbar.update(1)
        # break
