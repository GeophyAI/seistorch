"""Perform full waveform inversion."""
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
# torch.set_float32_matmul_precision('high')

import argparse
import os
import pickle
import socket
import time
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import setproctitle
import torch
import tqdm
from mpi4py import MPI
from mpi4py.util import pkl5
from torch.utils.tensorboard import SummaryWriter
from yaml import dump, load

import seistorch
from seistorch.eqconfigure import Shape
from seistorch.distributed import task_distribution_and_data_reception
from seistorch.io import SeisIO, DataLoader
from seistorch.log import SeisLog
from seistorch.signal import SeisSignal
from seistorch.model import build_model
from seistorch.type import TensorList
from seistorch.setup import *
from seistorch.utils import (DictAction, to_tensor)

from ot.utils import proj_simplex

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
parser.add_argument('--grad-cut', action='store_true',
                    help='Cut the boundaries of gradient or not')
parser.add_argument('--grad-smooth', action='store_true',
                    help='Smooth the gradient or not')
parser.add_argument('--source-encoding', action='store_true', default=False,
                    help='PLEASE DO NOT CHANGE THE DEFAULT VALUE.')

if __name__ == '__main__':
    args = parser.parse_args()

    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()

    seislog = SeisLog(backend="MPI")

    seislog.print("Configuration: %s" % args.config)

    args.dev = setup_device(rank, args.use_cuda)    
 
    'Sets the number of threads used for intraop parallelism on CPU.'
    torch.set_num_threads(args.num_threads)

    # Build model
    cfg, model = build_model(args.config, device=args.dev, mode=args.mode, source_encoding=args.source_encoding, commands=args, logger=seislog)
    # model = torch.compile(model)
    seisio = SeisIO(cfg)
    seissignal = SeisSignal(cfg, seislog)
    setup = SeisSetup(cfg, args, seislog)
    setup.setup_seed()
    # Rank 0 is the master node for assigning tasks
    MASTER = rank == 0
    # Rank 1 is the writter node for writing logs
    WRITTER = rank == 1

    # Set the name of the process
    proc_name = cfg['name'] if not MASTER else "TaskAssign"
    setproctitle.setproctitle(proc_name)

    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list = seisio.read_geom(cfg)

    """Short cuts of the configures"""
    NSHOTS = setup.setup_num_shots()
    cfg['geom']['Nshots'] = NSHOTS

    # shape must be defined after NSHOTS is updated
    shape = Shape(cfg)

    use_mpi = size > 1

    if (use_mpi and rank!=0) or (not use_mpi):
        model.to(args.dev)
        model.train()

        # Set up the wavelet
        x = setup.setup_wavelet()

    """---------------------------------------------"""
    """-------------------MODELING------------------"""
    """---------------------------------------------"""
    
    if args.mode == 'forward':

        if MASTER:
            # in case each record have different shape
            record = setup.setup_file_system()
            # record = np.empty(NSHOTS, dtype=np.ndarray) 
        else:
            x = torch.unsqueeze(x, 0)

        comm.Barrier()
        # Rank 0 is the master node for assigning tasks
        if MASTER:
            num_batches = min(NSHOTS, args.num_batches)
            pbar = setup.setup_pbar(num_batches, cfg['equation'])
            shots = setup.setup_tasks()
            kwargs = {'record': record}
            
            #shots = np.array([180])
        
            task_distribution_and_data_reception(shots, pbar, args.mode, num_batches, **kwargs)

        else:
            # Other ranks are the worker nodes
            while True:
                # receive task from the master node
                tasks = comm.recv(source=0, tag=MPI.ANY_TAG)

                # break the loop if the master node has sent stop signal
                if tasks == -1:
                    break

                # Forward modeling
                with torch.no_grad():
                    shots = tasks
                    model.reset_geom(shots, src_list, rec_list, cfg)
                    model.cell.geom.step() # random boundary
                    y = model(x)
                    record = y.numpy()
                
                comm.send((tasks, rank, record), dest=0, tag=1)

        comm.Barrier()

        """Save the modeled data, which stored in rank0 <record>"""
        if MASTER:
            pbar.close()
            record.write()
            # fp = np.memmap(cfg['geom']['obsPath'], dtype=np.ndarray, mode='r')
            # for i in range(record.shape[0]):
                # print(i, fp[i].shape, record[i].shape)
            seislog.print("Modeling done, data will be writing to disk.")
            #seisio.to_file(cfg['geom']['obsPath'], record)

    """---------------------------------------------"""
    """-------------------INVERSION-----------------"""
    """---------------------------------------------"""

    if args.mode in ['inversion', 'rtm']:
        epoch_per_scale = cfg['training']['N_epochs']
        ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]
        MINIBATCH = cfg['training']['minibatch']
        IMPLICIT = cfg['training']['implicit']['use']
        MULTISCALES = cfg['geom']['multiscale']
        num_scales = len(MULTISCALES)
        NDIM = model.cell.geom.ndim
        cfg["loss"] = args.loss
        cfg["ROOTPATH"] = ROOTPATH
        cfg['training']['lr'] = args.lr
        cfg['training']['optimizer'] = args.opt
        cfg['gradient_cut'] = args.grad_cut
        cfg['gradient_smooth'] = args.grad_smooth

        BATCHSIZE, num_batches = setup.setup_batchsize()
        SEABED = setup.setup_seabed()

        if "obsmask" in cfg["geom"].keys():
            obsmask = DataLoader(cfg["geom"]["obsmask"])
            # obsmask = seisio.fromfile(cfg["geom"]["obsmask"])
            
        if "synmask" in cfg["geom"].keys():
            synmask = DataLoader(cfg["geom"]["synmask"])
            # synmask = seisio.fromfile(cfg["geom"]["synmask"])

        if MASTER:
            seislog.print(f"Working in dimension {NDIM}")
            os.makedirs(ROOTPATH, exist_ok=True)
            seisio.write_cfg(f"{ROOTPATH}/configure.yml", cfg)

        if WRITTER:
            writer = SummaryWriter(os.path.join(ROOTPATH, "logs"))

        """Define the misfit function"""
        criterions = setup.setup_criteria()

        # For traditional inversion, A GRAD with the same shape of the model
        # is used for reduce all the gradients to the master node   
        if not IMPLICIT:     
            GRAD = np.zeros(shape.grad_worker, np.float32)
 
        if MASTER:
            shots_this_iter = setup.setup_tasks()
            seislog.print(f"loss: {criterions}")

            obs0 = setup.setup_file_system()
            comm.bcast(obs0, root=0)

            loss = np.zeros((num_scales, epoch_per_scale, NSHOTS), np.float32)

        else:
            # The gradient of other ranks are 2D arrays.
            obs0 = comm.bcast(None, root=0)      


        """Loop over all scale"""
        for epoch in range(epoch_per_scale*num_scales):
            
            idx_freq, local_epoch = divmod(epoch, epoch_per_scale)
            model.cell.geom.step(SEABED)

            if IMPLICIT:
                gradient_dict = {}
            else:
                GRAD.fill(0.)

            if local_epoch==0:

                """Filter the data at every scale"""
                freq = MULTISCALES[idx_freq]
                if MASTER:
                    seislog.print(f"Current scale: {freq} Hz")
                    # Filter both record and ricker
                    #filtered_data = seissignal.filter(full_band_data, freqs=freq)
                    # Pickle the filtered data
                    #data_str = pickle.dumps(filtered_data)
                    pass
                # Broadcast the filtered data to other processors
                if MASTER:
                    pass
                    #comm.bcast(data_str, root=0)
                else:
                    pass
                    #data_str = comm.bcast(None, root=0)
                    #filtered_data = pickle.loads(data_str)

                # Reset the optimizer at each scale
                optimizers, lr_scheduler = setup.setup_optimizer(model, idx_freq, IMPLICIT)
                if (use_mpi and not MASTER) or (not use_mpi):
                    # Low pass filtered wavelet
                    if isinstance(x, torch.Tensor): x = x.numpy()
                    # lp_wavelet = seissignal.filter(x.copy().reshape(1, -1), freqs=freq)[0]
                    # the wavelet will not be filter.
                    # we will filter the synthetic data at each epoch
                    lp_wavelet = seissignal.filter(x.copy().reshape(1, -1), freqs='all')[0]
                    lp_wavelet = torch.unsqueeze(torch.from_numpy(lp_wavelet), 0)
        
            """Loop over all epoches"""
            # Master rank will assign tasks to other ranks
            if MASTER:

                pbar = setup.setup_pbar(num_batches, f"E{local_epoch+1}/{epoch_per_scale} | F{idx_freq+1}/{num_scales}")

                shots = next(shots_this_iter)#np.random.choice(np.arange(NSHOTS), BATCHSIZE, replace=False) if MINIBATCH else np.arange(NSHOTS)
                # seislog.print(f"shots: {shots}")
                kwargs = {'loss': loss, 
                          'epoch': local_epoch, 
                          'grad3d': None,
                          'idx_freq': idx_freq, 
                          'ndim': NDIM, 
                          'ROOTPATH': ROOTPATH}
                
                task_distribution_and_data_reception(shots, pbar, args.mode, num_batches, **kwargs)

            else:
                while True:
                    # Receive task from the master node
                    tasks = comm.recv(source=0, tag=MPI.ANY_TAG)
                    # Break the loop if the master node has sent stop signal
                    if tasks == -1: break
                    shots_this_rank = tasks

                    """Calculate one shot gradient"""
                    def closure():
                        optimizers.zero_grad()
                        """Although it is a for loop """
                        """But only one shot here when traditional workflow is using"""
                        model.reset_geom(shots_this_rank, src_list, rec_list, cfg)
                        syn = model(lp_wavelet) # syn is a TensorList
                        # Filter at each epoch
                        fobs = seissignal.filter(obs0[shots_this_rank], freqs=freq)
                        obs = TensorList(fobs.tolist()).to(syn.device)
                        # Filter the syn data
                        syn = seissignal.filter(syn, freqs=freq, backend='torch')
                        # FOR RTM
                        syn = syn.stack()
                        obs = obs.stack()

                        if "obsmask" in cfg["geom"].keys() and "synmask" in cfg["geom"].keys():
                            obsM = to_tensor(np.stack(obsmask[shots_this_rank], axis=0)).to(syn.device)
                            synM = to_tensor(np.stack(synmask[shots_this_rank], axis=0)).to(syn.device)
                            syn = syn * synM
                            obs = obs * obsM

                        #if shot==10:
                        #name_postfix = 'init' if epoch==0 else ''
                        name_postfix = ''
                        #np.save(f"{ROOTPATH}/obs{name_postfix}.npy", obs.cpu().detach().numpy())
                        #np.save(f"{ROOTPATH}/syn{name_postfix}.npy", syn.cpu().detach().numpy())
                        loss = criterions(syn, obs)
                        #adj = torch.autograd.grad(loss, syn, create_graph=True)[0]
                        #np.save(f"{ROOTPATH}/adj.npy", adj.detach().cpu().numpy())        
                        if IMPLICIT:
                            torch.save(model.cell.geom.nn['vp'].state_dict(), 
                                    f"{ROOTPATH}/nn{rank}.pt")
                        # For random boundary
                        # model.cell.geom.step()

                        loss.backward()

                        return loss.item()

                    # Run the closure
                    loss = closure()

                    """Sum the gradients of all the batches"""

                    if not IMPLICIT:
                        for idx, mname in enumerate(model.cell.geom.pars_need_invert):
                            GRAD[idx][:]+=getattr(model.cell.geom, mname).grad.cpu().detach().numpy()
                        # GRAD[idx][:]+=model.cell.geom.__getattr__(mname).grad.cpu().detach().numpy()
                    
                    if IMPLICIT:
                        for name, param in model.cell.geom.nn['vp'].named_parameters():
                            if param.grad is not None:
                                gradient_dict[name] = param.grad.clone().detach()

                    # Send to the master node
                    comm.send((tasks, rank, None, loss), dest=0, tag=1)

            comm.Barrier()

            if not IMPLICIT:
                # SUM THE GRADIENTS FROM ALL RANKS
                GRAD = comm.allreduce(GRAD, op=MPI.SUM)

            if IMPLICIT:
                all_nn = comm.gather(gradient_dict, root=0)

            """"Assigning and Saving"""
            if MASTER:
                pbar.close()
                np.save(f"{ROOTPATH}/loss.npy", loss)
                if not IMPLICIT:
                    np.save(f"{ROOTPATH}/grad.npy", GRAD)
                
                if IMPLICIT:
                    all_nn = all_nn[1:]
                    # copy grad to data
                    for name, param in model.cell.geom.nn['vp'].named_parameters():
                        grad = torch.sum(torch.stack([nn[name].cpu() for nn in all_nn]), axis=0)
                        param.data.copy_(grad.to(args.dev)/1e6)

                    # Save the sumemd gradients
                    torch.save(model.cell.geom.nn['vp'].state_dict(), f'{ROOTPATH}/grad.pt')

                # print(nnfiles)
            if not MASTER: # Post processing for gradients

                if IMPLICIT:
                    summed_grad = torch.load(f'{ROOTPATH}/grad.pt', map_location='cpu')
                    for name, param in model.cell.geom.nn['vp'].named_parameters():
                        param.grad = summed_grad[name].to(args.dev)

                # Assign gradient of other ranks
                if not IMPLICIT:
                    for idx, para in enumerate(model.cell.geom.pars_need_invert):
                        var = model.cell.geom.__getattr__(para)
                        var.grad.data = to_tensor(GRAD[idx]).to(args.dev)

                if args.grad_smooth:
                    model.cell.geom.gradient_smooth()

                if args.grad_cut and isinstance(SEABED, torch.Tensor):
                    model.cell.geom.gradient_cut(SEABED, cfg['geom']['pml']['N'])        
                    
                #torch.nn.utils.clip_grad_norm_(model.cell.parameters(), 1e-2)
                # Update the model parameters and learning rate
                optimizers.step()
                lr_scheduler.step()
                # model.cell.geom.step()
                # Proj simplex
                # if True:
                #     for idx, para in enumerate(model.cell.geom.pars_need_invert):
                #         var = model.cell.geom.__getattr__(para)
                #         var.data = proj_simplex(var)

            if WRITTER:
                # Save vel and grad
                model.cell.geom.save_model(ROOTPATH, 
                                           paras=["vel", "grad"], 
                                           freq_idx=idx_freq,
                                           writer=writer,
                                           epoch=local_epoch)
