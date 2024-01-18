"""Perform full waveform inversion."""
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True

import argparse
import os
import pickle
import socket
import time

import numpy as np
import setproctitle
import torch
import tqdm
import torch.distributed as dist
from yaml import dump, load

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import seistorch
from seistorch.eqconfigure import Shape
from seistorch.distributed import task_distribution_and_data_reception
from seistorch.io import SeisIO, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from seistorch.log import SeisLog
from seistorch.signal import SeisSignal
from seistorch.source import WaveSource
from seistorch.probe import WaveProbe
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.utils import (DictAction, to_tensor)
from seistorch.utils import merge_sources_with_same_keys, merge_receivers_with_same_keys
from seistorch.setup import *
from seistorch.dataset import OBSDataset
from seistorch.type import TensorList



parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('--num_threads', type=int, default=2,
                    help='Number of threads to use')
parser.add_argument('--num-batches', type=int, default=1,
                    help='Number of batches to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--opt', choices=['adam', 'lbfgs', 'steepestdescent', 'cg'], default='adam',
                    help='optimizer (adam)')
parser.add_argument('--save-path', default='',
                    help='the root path for saving results')
parser.add_argument('--loss', action=DictAction, nargs="+",
                    help='loss dictionary')
parser.add_argument('--lr', action=DictAction, nargs="+",
                    help='learning rate')
parser.add_argument('--mode', choices=['forward', 'inversion', 'rtm'], default='forward',
                    help='forward modeling, inversion or reverse time migration mode')
parser.add_argument('--modelparallel', action='store_true',
                    help='Split the model to various GPUs')
parser.add_argument('--grad-cut', action='store_true',
                    help='Cut the boundaries of gradient or not')
parser.add_argument('--grad-smooth', action='store_true',
                    help='Smooth the gradient or not')
parser.add_argument('--source-encoding', action='store_true', default=False,
                    help='PLEASE DO NOT CHANGE THE DEFAULT VALUE.')

if __name__ == '__main__':

    dist.init_process_group("nccl")

    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    size = int(os.environ['WORLD_SIZE'])

    MASTER = rank == 0

    dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
   
    # Build model
    cfg, model = build_model(args.config, device=dev, mode='inversion', source_encoding=False, commands=args)

    # Send to GPU
    model.to(rank)

    seislog = SeisLog(backend="TORCHRUN")
    seisio = SeisIO(cfg)
    seissignal = SeisSignal(cfg, seislog)
    setup = SeisSetup(cfg, args, seislog)
    
    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list = seisio.read_geom(cfg)

    # Setup wavelet
    x = setup.setup_wavelet().to(rank)
    x = torch.unsqueeze(x, 0)

    model = DistributedDataParallel(model, device_ids=[rank])
    seislog.print('DistributedDataParallel is used.')

    """CONFIGURES"""
    EPOCH_PER_SCALE = cfg['training']['N_epochs']
    ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]
    MINIBATCH = cfg['training']['minibatch']
    MULTISCALES = cfg['geom']['multiscale']
    IMPLICIT = cfg['training']['implicit']['use']
    SCALE_COUNTS = len(MULTISCALES)

    # UPDATE THE CONFIGURATION FILE
    cfg['loss'] = args.loss
    cfg['ROOTPATH'] = ROOTPATH
    cfg['training']['lr'] = args.lr
    cfg['training']['optimizer'] = args.opt
    cfg['gradient_cut'] = args.grad_cut
    cfg['gradient_smooth'] = args.grad_smooth

    if MASTER:
        os.makedirs(ROOTPATH, exist_ok=True)
        seisio.write_cfg(f"{ROOTPATH}/configure.yml", cfg)
        # writer = SummaryWriter(os.path.join(ROOTPATH, "logs"))

    """Load obs data"""
    obs0 = OBSDataset(cfg['geom']['obsPath'], 
                      dkey='shot', 
                      srclist=src_list,
                      reclist=rec_list,
                      freqs=None, 
                      PMLN=cfg['geom']['pml']['N'], 
                      MULTIPLE=cfg['geom']['multiple'])

    SHOTS_PER_BATCH = 4 # USE SHOTS_PER_BATCH for GRADIENT
    batch_size_per_gpu = SHOTS_PER_BATCH//size
    obssampler = DistributedSampler(obs0)
    obsloader = torch.utils.data.DataLoader(obs0, 
                                            batch_size=batch_size_per_gpu, 
                                            shuffle=False, 
                                            sampler=obssampler)

    seislog.print(f'Obs data is loaded. {len(obs0)} shots in total.')

    """Define the misfit function"""
    criterions = setup.setup_criteria()

    obsmask, synmask = setup.setup_datamask()

    """Tranining"""
    for epoch in tqdm.trange(EPOCH_PER_SCALE*SCALE_COUNTS):

        idx_freq, local_epoch = divmod(epoch, EPOCH_PER_SCALE)

        if local_epoch==0:
            freq = MULTISCALES[idx_freq]
            optimizers, lr_scheduler = setup.setup_optimizer(model, idx_freq, IMPLICIT)


        optimizers.zero_grad()
        
        obssampler.set_epoch(epoch)
        obs, src, rec = next(iter(obsloader)) # in grids, no padding

        """Processing the coordinates"""
        rec = [torch.stack(item) for item in rec]
        rec = torch.stack(rec).permute(2, 0, 1).cpu().numpy().tolist()

        src = torch.stack(src).cpu().numpy().T.tolist()

        padded_src, padded_rec = setup_acquisition2(src, rec, cfg)

        if isinstance(padded_src, list):
            padded_src = torch.nn.ModuleList(padded_src)
        else:
            padded_src = torch.nn.ModuleList([padded_src])

        if isinstance(padded_rec, list):
            padded_rec = torch.nn.ModuleList(padded_rec)
        else:
            padded_rec = torch.nn.ModuleList([padded_rec])

        # Get the super source and super probes
        bidx_source, sourcekeys = merge_sources_with_same_keys(padded_src)
        super_source = WaveSource(bidx_source, **sourcekeys).to(dev) 

        reccounts, bidx_receivers, reckeys = merge_receivers_with_same_keys(padded_rec)
        super_probes = WaveProbe(bidx_receivers, **reckeys).to(dev)

        syn = model(x, None, super_source, super_probes)
        obs = TensorList(obs).to(dev)

        """Filter the data"""
        syn = seissignal.filter(syn, freqs=freq, backend='torch')
        obs = seissignal.filter(obs, freqs=freq, backend='torch')

        loss = criterions(syn, obs)

        # SAVE THE DATA
        torch.save(obs, f"{ROOTPATH}/obs_{epoch}.pt")
        torch.save(syn, f"{ROOTPATH}/syn_{epoch}.pt")

        loss.backward()
        optimizers.step()
        lr_scheduler.step()

        if MASTER:
            # SAVE THE INVERTED MODEL
            torch.save(model.module.state_dict(), 
                       f"{ROOTPATH}/model_{epoch}.pt")
            # SAVE THE GRADIENT
            torch.save(model.module.cell.geom.vp.grad, 
                       f"{ROOTPATH}/grad_{epoch}.pt")

        # print(loss)



    



