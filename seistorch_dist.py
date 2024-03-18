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
from torch.utils.data.distributed import DistributedSampler
import seistorch
from seistorch.eqconfigure import Shape
from seistorch.distributed import task_distribution_and_data_reception
from seistorch.io import SeisIO, DataLoader
from torch.utils.tensorboard import SummaryWriter
from seistorch.log import SeisLog
from seistorch.coords import single2batch
from seistorch.signal import SeisSignal, generate_arrival_mask
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.utils import (DictAction, to_tensor)
from seistorch.setup import *
from seistorch.dataset import OBSDataset
from seistorch.type import TensorList
from seistorch.process import PostProcess
from torch.distributed.elastic.multiprocessing.errors import record


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
parser.add_argument('--grad-clip', action='store_true', default=True,
                    help='Clip the gradient or not')
parser.add_argument('--filteratfirst', action='store_true', 
                    help='Filter the wavelet at the first step or not')
parser.add_argument('--clipvalue', type=float, default=0.02)
parser.add_argument('--source-encoding', action='store_true', default=False,
                    help='PLEASE DO NOT CHANGE THE DEFAULT VALUE.')


if __name__ == "__main__":

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
    postprocess = PostProcess(model, cfg, args)
    
    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list = seisio.read_geom(cfg)

    # Setup wavelet
    x = setup.setup_wavelet().to(rank)

    x = torch.unsqueeze(x, 0)

    """CONFIGURES"""
    EPOCH_PER_SCALE = cfg['training']['N_epochs']
    ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]
    MINIBATCH = cfg['training']['minibatch']
    MULTISCALES = cfg['geom']['multiscale']
    IMPLICIT = cfg['training']['implicit']['use']
    SCALE_COUNTS = len(MULTISCALES)
    SHOTS_PER_BATCH = cfg['training']['batch_size'] # USE SHOTS_PER_BATCH for GRADIENT
    
    if IMPLICIT:
        nn = DistributedDataParallel(model.cell.geom.nn['vp'], device_ids=[rank])
    else:
        model = DistributedDataParallel(model, device_ids=[rank])
    seislog.print('DistributedDataParallel is used.')

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
        pbar = tqdm.tqdm(total=EPOCH_PER_SCALE*SCALE_COUNTS)
        writer = SummaryWriter(os.path.join(ROOTPATH, "logs"))

    """Load obs data"""
    obs0 = OBSDataset(cfg['geom']['obsPath'], 
                      dkey='shot', 
                      srclist=src_list,
                      reclist=rec_list,
                      freqs=None, 
                      PMLN=cfg['geom']['boundary']['width'], 
                      MULTIPLE=cfg['geom']['multiple'])

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
    usemask = obsmask is not None or synmask is not None

    """Tranining"""
    for epoch in range(EPOCH_PER_SCALE*SCALE_COUNTS):
        
        idx_freq, local_epoch = divmod(epoch, EPOCH_PER_SCALE)

        if local_epoch==0:
            """Reset the optimizer at every scale"""
            freq = MULTISCALES[idx_freq]
            optimizers, lr_scheduler = setup.setup_optimizer(model, 
                                                             idx_freq, 
                                                             IMPLICIT, 
                                                             args.grad_clip, 
                                                             args.clipvalue)

        if args.filteratfirst: 
            lpx = seissignal.filter(x.cpu().numpy().copy().reshape(1, -1), freqs=freq)[0]
        else:
            lpx = seissignal.filter(x.cpu().numpy().copy().reshape(1, -1), freqs='all')[0]
        lpx = torch.unsqueeze(torch.from_numpy(lpx), 0)

        optimizers.zero_grad()
        
        obssampler.set_epoch(epoch)
        """Get the data"""
        obs, src, rec, shots = next(iter(obsloader)) # in grids, no padding
        super_source, super_probes = single2batch(src, rec, cfg, dev) # padding, in batch

        if IMPLICIT:
            coords = model.cell.geom.nn['vp'].coords
            vp = nn(coords)[0]
            std, mean = 1000., 3000.
            vp = vp * std + mean
            # vp[0,:] = 1500.
        else:
            vp = None

        """Forward modeling"""
        syn = model(lpx, None, super_source, super_probes, vp=vp)
        obs = TensorList(obs).to(dev)

        """Filter the data"""
        if not args.filteratfirst:
            syn = seissignal.filter(syn, freqs=freq, backend='torch')
        obs = seissignal.filter(obs, freqs=freq, backend='torch')

        """Apply the mask"""
        obs = obs.stack()
        syn = syn.stack()

        np.save(f'{ROOTPATH}/syn_nomask_{rank}.npy', syn.cpu().detach().numpy())
        np.save(f'{ROOTPATH}/obs_nomask_{rank}.npy', obs.cpu().detach().numpy())

        if usemask:
            if obsmask is not None:
                obsM = to_tensor(np.stack(obsmask[shots.cpu().numpy().tolist()], axis=0)).to(syn.device)
                obs = obs * obsM

            if synmask is not None:
                synM = to_tensor(np.stack(synmask[shots.cpu().numpy().tolist()], axis=0)).to(syn.device)
            else:
                synM = generate_arrival_mask(syn, top_win=200, down_win=200)
            
            syn = syn * synM


        """Compute the loss"""
        loss = criterions(syn, obs)

        np.save(f'{ROOTPATH}/syn{rank}.npy', syn.cpu().detach().numpy())
        np.save(f'{ROOTPATH}/obs{rank}.npy', obs.cpu().detach().numpy())

        # adj = torch.autograd.grad(loss, syn)[0]
        # np.save(f'{ROOTPATH}/adj{rank}.npy', adj.cpu().detach().numpy())

        loss.backward()

        if MASTER:
            if not IMPLICIT:
                # SAVE THE GRADIENT
                for par in model.module.cell.geom.pars_need_invert:
                    tensor = model.module.cell.geom.__getattr__(par).grad
                    torch.save(tensor, 
                            f"{ROOTPATH}/grad_{par}_nosm_{epoch}.pt")

        """Post-processing"""
        if args.grad_smooth:
            postprocess.smooth_gradient()

        if args.grad_cut:
            postprocess.cut_gradient()

        if False:
            postprocess.repad()

        optimizers.step()
        lr_scheduler.step()

        if MASTER:
            # SAVE THE INVERTED MODEL
            if IMPLICIT:
                torch.save(vp, 
                           f"{ROOTPATH}/model_{epoch}.pt")
            else:
                torch.save(model.module.state_dict(), 
                           f"{ROOTPATH}/model_{epoch}.pt")
                # SAVE THE GRADIENT
                for par in model.module.cell.geom.pars_need_invert:
                    tensor = model.module.cell.geom.__getattr__(par).grad
                    torch.save(tensor, 
                            f"{ROOTPATH}/grad_{par}_{epoch}.pt")
            pbar.update(1)

            writer.add_histogram('Sample Indices', shots, epoch)
            writer.add_scalar('Loss', loss, epoch)

