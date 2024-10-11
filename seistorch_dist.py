"""Perform full waveform inversion."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch._dynamo.config.capture_scalar_outputs = True

import os

import numpy as np
import setproctitle
import torch
import tqdm
import torch.distributed as dist
from yaml import dump, load

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import seistorch
from seistorch.io import SeisIO
from torch.utils.tensorboard import SummaryWriter
from seistorch.log import SeisLog
from seistorch.coords import single2batch, offset_with_boundary
from seistorch.signal import SeisSignal, generate_arrival_mask
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.utils import (to_tensor, nestedlist2tensor)
from seistorch.dataset import OBSDataset
from seistorch.array import TensorList
from seistorch.process import PostProcess
from seistorch.parser import fwi_parser as parser


if __name__ == "__main__":

    dist.init_process_group("nccl")

    args = parser().parse_args()
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
    SHOTS_PER_EPOCH = cfg['training']['batch_size'] # USE SHOTS_PER_EPOCH for GRADIENT
    
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

    STEP_PER_EPOCH = args.step_per_epoch

    if MASTER:
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

    SHOTS_PER_GPU = SHOTS_PER_EPOCH//size
    obssampler = DistributedSampler(obs0)
    obsloader = torch.utils.data.DataLoader(obs0, 
                                            batch_size=SHOTS_PER_GPU, 
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
            if MASTER: pbar.reset()
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

        # if EPOCH_IS_START:
        optimizers.zero_grad()
        
        obssampler.set_epoch(epoch)
        """Get the data"""
        _obs, _src, _rec, _shots = next(iter(obsloader)) # in grids, no padding

        bps = SHOTS_PER_GPU//STEP_PER_EPOCH # batchsize_per_step

        assert bps > 0, f"Num. of tasks per GPU is {SHOTS_PER_GPU}, but step per epoch is {STEP_PER_EPOCH}."
        loss_all_batch = 0.
        # Loop over the batch
        for bps_idx in range(STEP_PER_EPOCH):
            start = bps_idx*bps
            end = (bps_idx+1)*bps

            obs = _obs[start:end]
            src = [s[start:end] for s in _src]
            rec = nestedlist2tensor(_rec)[...,start:end] # (ncoords, nrecs, nshots)
            shots = _shots[start:end]

            src = to_tensor(src).T
            rec = to_tensor(rec).permute(2, 0, 1)

            src, rec = offset_with_boundary(src, rec, cfg)

            batched_source, batched_probes = single2batch(src, rec, cfg, dev) # padding, in batch

            if IMPLICIT:
                coords = model.cell.geom.nn['vp'].coords
                vp = nn(coords)[0]
                std, mean = 1000., 3000.
                vp = vp * std + mean
                # vp[0,:] = 1500.
            else:
                vp = None

            """Forward modeling"""
            syn = model(lpx, None, batched_source, batched_probes, vp=vp)
            obs = TensorList(obs).to(dev)

            """Filter the data"""
            if not args.filteratfirst:
                syn = seissignal.filter(syn, freqs=freq, backend='torch')
            if not args.obsnofilter:
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
                # for special use
                # synM = generate_arrival_mask(syn, top_win=200, down_win=200)
                # syn = syn * synM

            """Compute the loss"""
            loss = criterions(syn, obs)

            np.save(f'{ROOTPATH}/syn{rank}.npy', syn.cpu().detach().numpy())
            np.save(f'{ROOTPATH}/obs{rank}.npy', obs.cpu().detach().numpy())

            # adj = torch.autograd.grad(loss, syn)[0]
            # np.save(f'{ROOTPATH}/adj{rank}.npy', adj.cpu().detach().numpy())

            loss.backward()
            loss_all_batch += loss.item()

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

        if args.source_illumination:
            postprocess.precondition()

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
            # if EPOCH_IS_DONE:
            pbar.update(1)
            # pbar_in_epoch.n = 0
            # pbar_in_epoch.update(1)
            writer.add_histogram('Sample Indices', _shots, epoch)
            writer.add_scalar('Loss', loss_all_batch, epoch)

