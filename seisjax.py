"""Perform full waveform inversion."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding"

import jax
import optax
from jax import pmap

import argparse

import numpy as np
import tqdm
# import torch.distributed as dist
from yaml import dump, load
from functools import partial

import seistorch
from seistorch.io import SeisIO
from torch.utils.tensorboard import SummaryWriter
from seistorch.array import SeisArray
from seistorch.log import SeisLog
from seistorch.coords import single2batch, offset_with_boundary
from seistorch.signal import SeisSignal
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.dataset import OBSDataset, NumpyLoader, DataLoaderJAX
from seistorch.process import PostProcess
from seistorch.parser import fwi_parser as parser
from seistorch.utils import inplace_update, inplace_zeros

if __name__ == "__main__":

    args = parser().parse_args()

    # Sharding across devices
    devices = jax.devices()
    devices_count = len(devices)
    mesh = jax.sharding.Mesh(np.array(devices), ('devices',))
    input_spec = jax.sharding.PartitionSpec('devices', )
    dist_sharding = jax.sharding.NamedSharding(mesh, input_spec)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    print('Executing on devices:', devices)
    # Build model
    cfg, model = build_model(args.config, device=str(devices), mode='inversion', source_encoding=False, commands=args, sharding=replicated_sharding)

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

    SHOTS_PER_GPU = SHOTS_PER_EPOCH#//devices_count

    bps = SHOTS_PER_GPU//STEP_PER_EPOCH # batchsize / step / gpu
    
    assert bps > 0, f"Num. of tasks per GPU is {SHOTS_PER_GPU}, but step per epoch is {STEP_PER_EPOCH}."
    
    loss_all_batch = 0.

    obsloader = DataLoaderJAX(obs0, batch_size=bps, shuffle=False, sampler=None)

    # initial paras
    local_params = model.parameters()
    dist_params = pmap(lambda _: model.parameters())(jnp.arange(devices_count)) # (ndevices, nparams, nz, nx, ...)

    rng_key = pmap(lambda x: jax.random.PRNGKey(cfg['seed']+x))(jnp.arange(devices_count))
    criterions = setup.setup_criteria()

    x = jax.device_put(x, replicated_sharding)

    @partial(jax.jit, static_argnames=('freqs', ))
    def step(obs, src , rec, rng_key, params, freqs):

        rng_key, _ = jax.random.split(rng_key)
        src, rec = offset_with_boundary(src, rec, cfg)
        # Reset the parameters
        model.set_parameters(params) # Optax-based updates
        batched_source, batched_probes = single2batch(src, rec, cfg, 'cpu') # padding, in batch
        obs = SeisArray(obs).filter(cfg['geom']['dt'], freqs, FORDER, axis=1)

        def loss(params):
            syn = model(x, None, batched_source, batched_probes, parameters=params)
            # apply filter
            syn = SeisArray(syn).filter(cfg['geom']['dt'], freqs, FORDER, axis=1)
            assert obs.shape == syn.shape, f"Observed shape {obs.shape} is not equal to synthetic shape {syn.shape}"
            return criterions(syn, obs), syn

        def compute_gradient(params):
            return jax.value_and_grad(loss, has_aux=True)(params)

        (_loss, syn), gradient = compute_gradient(params)

        return _loss, params, rng_key, gradient, syn, obs, shots

    batch_gradients =jnp.zeros((len(local_params), *local_params[0].shape), dtype=jnp.float32) # (nparams, ndevices, nz, nx, ...)

    step = pmap(step, axis_name='devices', static_broadcasted_argnums=(5,))

    @jax.jit
    def get_data_batch(key):

        obsloader.reset_key(key)
        obs, src, rec, shots = next(iter( obsloader ))

        local_shape = (devices_count, obs.shape[0]//devices_count)

        obs = obs.reshape(local_shape+ obs.shape[1:])
        src = src.reshape(local_shape+ src.shape[1:])
        rec = rec.reshape(local_shape+ rec.shape[1:])

        obs = jax.device_put(obs, dist_sharding)
        src = jax.device_put(src, dist_sharding)
        rec = jax.device_put(rec, dist_sharding)

        return obs, src, rec, shots

    # Loop over the epochs
    for epoch in range(EPOCH_PER_SCALE*SCALE_COUNTS):

        loss_all_batch = 0.
        idx_freq, local_epoch = divmod(epoch, EPOCH_PER_SCALE)

        if local_epoch==0:
            opt = setup.setup_optimizer_jax(idx_freq=idx_freq)
            opt_state = opt.init(model.parameters())
            pbar.reset()
        
        batch_gradients = inplace_zeros(batch_gradients)
        # Gradient accumulation
        for batch_step in range(STEP_PER_EPOCH):
            
            obs, src, rec, shots = get_data_batch(rng_key[0])
            loss_per_batch, dist_params, rng_key, gradient, syn, obs, shots = step(obs, src, rec, rng_key, dist_params, MULTISCALES[idx_freq])
            gradient_avg = []
            for grad in gradient:
                gradient_avg.append(jnp.mean(grad, axis=0))

            batch_gradients = inplace_update(batch_gradients, gradient_avg)
            loss_all_batch += loss_per_batch

        # Update the parameters
        updates, opt_state = opt.update(tuple(batch_gradients), opt_state)

        local_params = optax.apply_updates(local_params, updates)

        dist_params = pmap(lambda _: local_params)(jnp.arange(devices_count))

        np.save(f"{ROOTPATH}/inverted{epoch:03d}.npy", local_params)
        np.save(f"{ROOTPATH}/gradient{epoch:03d}.npy", batch_gradients)
        writer.add_scalar('Loss', loss_all_batch.sum().item(), epoch)

        # filtering = lambda path, value: isinstance(value, jnp.ndarray)
        # learning_rates = optax.tree_utils.tree_get_all_with_path(opt_state, 'learning_rate', filtering=filtering)
        # learning_rate = optax.tree_utils.tree_get( opt_state, 'learning_rate', filtering=filtering)
        # writer.add_scalar("Learning Rate", learning_rate.item(), epoch)
        # np.save(f"{ROOTPATH}/syn{epoch:03d}.npy", syn)
        # np.save(f"{ROOTPATH}/obs{epoch:03d}.npy", obs)
        pbar.update(1)
        
        # break