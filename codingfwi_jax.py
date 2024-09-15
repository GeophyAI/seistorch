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
from seistorch.coords import single2batch

from seistorch.eqconfigure import Shape
# from tensorflow.keras.models import load_model
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.log import SeisLog
from seistorch.io import SeisIO
from seistorch.signal import SeisSignal
from seistorch.utils import (DictAction, dict2table,
                             low_pass, roll_jax, to_tensor)
from seistorch.process import PostProcess
from seistorch.dataset import OBSDataset, NumpyLoader, RandomSampler, DataLoaderJAX
from seistorch.parser import coding_fwi_parser as parser
# parser = argparse.ArgumentParser()
# parser.add_argument('config', type=str, 
#                     help='Configuration file for geometry, training, and data preparation')
# parser.add_argument('--num_threads', type=int, default=2,
#                     help='Number of threads to use')
# parser.add_argument('--use-cuda', action='store_true',
#                     help='Use CUDA to perform computations')
# parser.add_argument('--gpuid', type=int, default=0,
#                     help='which gpu is used for calculation')
# parser.add_argument('--checkpoint', type=str,
#                     help='checkpoint path for resuming training')
# parser.add_argument('--opt', choices=['adam', 'lbfgs', 'cg', 'steepestdescent'], default='adam',
#                     help='optimizer (adam)')
# parser.add_argument('--loss', action=DictAction, nargs="+",
#                     help='loss dictionary')
# parser.add_argument('--save-path', default='',
#                     help='the root path for saving results')
# parser.add_argument('--lr', action=DictAction, nargs="+",
#                     help='learning rate')
# parser.add_argument('--batchsize', type=int, default=-1,
#                     help='batch size for coding')
# parser.add_argument('--grad-smooth', action='store_true',
#                     help='Smooth the gradient or not')
# parser.add_argument('--grad-cut', action='store_true',
#                     help='Cut the boundaries of gradient or not')
# parser.add_argument('--disable-grad-clamp', action='store_true',
#                     help='Clamp the gradient using quantile or not')
# parser.add_argument('--mode', choices=['inversion'], default='inversion',
#                     help='forward modeling, inversion or reverse time migration mode')
# parser.add_argument('--source-encoding', action='store_true', default=True,
#                     help='PLEASE DO NOT CHANGE THE DEFAULT VALUE.')
# parser.add_argument('--filteratlast', action='store_true', 
#                     help='Filter the wavelet at the last step or not')

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

    # TODO: add checkpoint
    # Resume training from checkpoint
    # assert os.path.exists(args.checkpoint), "Checkpoint not found"
    # Load the checkpoint
    # checkpoint = torch.load(args.checkpoint)

    # print("model state dict:", model.state_dict())
    # print("\n\n\n\n\n\n")
    # print(checkpoint)
    # # model.load_state_dict(checkpoint['model_state_dict'])
    # exit()
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
    probes = setup_rec_coords(full_rec_list, cfg['geom']['boundary']['width'], cfg['geom']['multiple'], use_jax=True)[0]
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
    # Optimizer for model parameters
    opt = setup.setup_optimizer_jax()
    opt_state = opt.init(model.parameters())

    # initial paras
    params = model.parameters()

    rng_key = jax.random.PRNGKey(20240908)

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
        _src = _src.T

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

        (_loss, coding_syn), gradient = compute_gradient(model.parameters())

        updates, opt_state = opt.update(gradient, opt_state)
        params = optax.apply_updates(model.parameters(), updates)

        return coding_obs, coding_syn, _shots, keys[0], params

    for epoch in range(EPOCH_PER_SCALE*SCALE_COUNTS):

        idx_freq, local_epoch = divmod(epoch, EPOCH_PER_SCALE)

        if local_epoch==0:
            pbar.reset()
        
        obs, syn, _shots, rng_key, params = step(epoch, opt_state, rng_key, params, freqs=MULTISCALES[idx_freq])
        np.save(f"{ROOTPATH}/model_F{idx_freq:02d}E{local_epoch:02d}.npy", params)
        pbar.set_description(f"F{idx_freq}E{local_epoch}")
        # np.save(f"{ROOTPATH}/obs_{epoch}.npy", obs)
        # np.save(f"{ROOTPATH}/syn_{epoch}.npy", syn)
        pbar.update(1)
        # break
