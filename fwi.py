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
from mpi4py import MPI
from mpi4py.util import pkl5
from torch.utils.tensorboard import SummaryWriter
from yaml import dump, load

import seistorch
from seistorch.eqconfigure import Shape
from seistorch.distributed import task_distribution_and_data_reception
from seistorch.io import SeisIO
from seistorch.signal import SeisSignal
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.utils import (DictAction, to_tensor)


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

    if args.use_cuda and torch.cuda.is_available():
        # Get the local_rank on each node
        torch.cuda.set_device(rank%torch.cuda.device_count())
        args.dev = torch.cuda.current_device()
        if rank ==0:
            print("Configuration: %s" % args.config)
            print("Using CUDA for calculation")
            print(f"Optimizer: {args.opt}")

    else:
        args.dev = torch.device('cpu')
        if rank ==0:
            print("Configuration: %s" % args.config)
            print("Using CPU for calculation")
    'Sets the number of threads used for intraop parallelism on CPU.'
    torch.set_num_threads(args.num_threads)
    # Build model

    cfg, model = build_model(args.config, device=args.dev, mode=args.mode, source_encoding=args.source_encoding, commands=args)

    seisio = SeisIO(cfg)
    seissignal = SeisSignal(cfg)
    shape = Shape(cfg)

    # Set the name of the process
    if rank!=0:
        setproctitle.setproctitle(cfg['name'])
    else:
        setproctitle.setproctitle("TaskAssign")

    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list = seisio.read_geom(cfg)

    """Short cuts of the configures"""
    EPOCHS = cfg['training']['N_epochs']
    NSHOTS = min(cfg['geom']['Nshots'], len(src_list))
    cfg['geom']['Nshots'] = NSHOTS
    use_mpi = size > 1
    if (use_mpi and rank!=0) or (not use_mpi):
        model.to(args.dev)
        model.train()

        # Set up the wavelet
        x = setup_wavelet(cfg)

    """---------------------------------------------"""
    """-------------------MODELING------------------"""
    """---------------------------------------------"""
    
    if args.mode == 'forward':

        if rank==0:
            # in case each record have different shape
            record = np.empty(NSHOTS, dtype=np.ndarray) 
        else:
            x = torch.unsqueeze(x, 0)

        comm.Barrier()
        # Rank 0 is the master node for assigning tasks
        if rank == 0:
            num_batches = min(NSHOTS, args.num_batches)
            pbar = tqdm.trange(num_batches, position=0, desc=cfg['equation'])
            shots = np.arange(NSHOTS)
            kwargs = {'record': record}

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
                    y = model(x)
                    record = y.cpu().detach().numpy()

                comm.send((tasks, rank, record), dest=0, tag=1)

        comm.Barrier()

        """Save the modeled data, which stored in rank0 <record>"""
        if rank==0:
            pbar.close()
            print("Modeling done, data will be writing to disk.")
            seisio.to_file(cfg['geom']['obsPath'], record)

    """---------------------------------------------"""
    """-------------------INVERSION-----------------"""
    """---------------------------------------------"""

    if args.mode in ['inversion', 'rtm']:

        """Write configure file to the inversion folder"""
        ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]
        MINIBATCH = cfg['training']['minibatch']
        BATCHSIZE = cfg['training']['batch_size'] if MINIBATCH else None
        cfg["loss"] = args.loss
        cfg["ROOTPATH"] = ROOTPATH
        cfg['training']['lr'] = args.lr
        cfg['training']['optimizer'] = args.opt
        cfg['gradient_cut'] = args.grad_cut
        cfg['gradient_smooth'] = args.grad_smooth
        
        SEABED = seisio.fromfile(cfg['geom']['seabed']) if 'seabed' in cfg['geom'].keys() else None
        #SEABED = np.load(cfg['geom']['seabed']) if 'seabed' in cfg['geom'].keys() else None
        SEABED = torch.from_numpy(SEABED).to(args.dev) if SEABED is not None else None
        
        if "datamask" in cfg["geom"].keys():
            datamask = seisio.fromfile(cfg["geom"]["datamask"])

        if rank==0:
            os.makedirs(ROOTPATH, exist_ok=True)
            seisio.write_cfg(f"{ROOTPATH}/configure.yml", cfg)

        if rank==1:
            writer = SummaryWriter(os.path.join(ROOTPATH, "logs"))

        """Define the misfit function"""
        # criterion = Loss(args.loss).loss(cfg)
        criterions = setup_criteria(cfg, args.loss)
        """Only rank0 will read the full band data"""
        """Rank0 will broadcast the data after filtering"""
        if rank == 0:
            full_band_data = seisio.fromfile(cfg['geom']['obsPath'])
            #filtered_data = np.zeros(shape.record3d, dtype=np.float32)
            loss = np.zeros((len(cfg['geom']['multiscale']), EPOCHS, NSHOTS), np.float32)
            # The gradient in rank0 is a 3D array.
            grad3d = np.zeros(shape.grad3d, np.float32)
            grad2d = np.zeros(shape.grad2d, np.float32)
        else:
            #filtered_data = np.zeros(shape.record3d, dtype=np.float32)
            # The gradient of other ranks are 2D arrays.
            grad2d = np.zeros(shape.grad2d, np.float32)
            #hessian = np.zeros(shape.hessian, np.float32)

        """Loop over all scale"""
        for idx_freq, freq in enumerate(cfg['geom']['multiscale']):

            if rank==0:
                # Filter both record and ricker
                filtered_data = seissignal.filter(full_band_data, freqs=freq)
                # Pickle the filtered data
                data_str = pickle.dumps(filtered_data)

            # Broadcast the filtered data to other processors
            if rank==0:
                comm.bcast(data_str, root=0)
            else:
                data_str = comm.bcast(None, root=0)
                filtered_data = pickle.loads(data_str)

            # Reset the optimizer at each scale
            optimizers, lr_scheduler = setup_optimizer(model, cfg, idx_freq)

            if (use_mpi and rank!=0) or (not use_mpi):
                # Low pass filtered wavelet
                if isinstance(x, torch.Tensor): x = x.numpy()
                lp_wavelet = seissignal.filter(x.copy().reshape(1, -1), freqs=freq)[0]
                lp_wavelet = torch.unsqueeze(torch.from_numpy(lp_wavelet), 0)

            """Loop over all epoches"""
            for epoch in range(EPOCHS):
                # Master rank will assign tasks to other ranks
                if rank == 0:
                    BATCHSIZE = min(NSHOTS, args.num_batches) if BATCHSIZE is None else BATCHSIZE
                    num_batches = min(NSHOTS, args.num_batches, BATCHSIZE)
                    pbar = tqdm.trange(num_batches, position=0)
                    shots = np.random.choice(np.arange(NSHOTS), BATCHSIZE, replace=False) if MINIBATCH else np.arange(NSHOTS)
                    num_tasks = shots.size
                    
                    # batched:
                    # shots = np.arange(NSHOTS)[epoch%BATCHSIZE:][::BATCHSIZE]
                    # shots = np.array([i*10 for i in range(8)])
                    #pbar = tqdm.tqdm(range(0, num_tasks), leave=False)
                    pbar.set_description(f"Freq{idx_freq}Epoch{epoch}")
                    kwargs = {'loss': loss, 
                              'epoch': epoch, 
                              'grad3d': grad3d,
                              'idx_freq': idx_freq}
                    
                    task_distribution_and_data_reception(shots, pbar, args.mode, num_batches, **kwargs)

                else:
                    while True:
                        # Receive task from the master node
                        tasks = comm.recv(source=0, tag=MPI.ANY_TAG)
                        # Break the loop if the master node has sent stop signal
                        if tasks == -1: break
                        shots = tasks

                        """Calculate one shot gradient"""
                        def closure():
                            optimizers.zero_grad()
                            """Although it is a for loop """
                            """But only one shot here when traditional workflow is using"""
                            model.reset_geom(shots, src_list, rec_list, cfg)
                            syn = model(lp_wavelet)
                            obs = to_tensor(np.stack(filtered_data[shots], axis=0)).to(syn.device)#.unsqueeze(0)

                            if "datamask" in cfg["geom"].keys():
                                dmask = to_tensor(np.stack(datamask[shots], axis=0)).to(syn.device)#.unsqueeze(0)
                                syn = syn * dmask
                                obs = obs * dmask

                            #if shot==10:
                            #name_postfix = 'init' if epoch==0 else ''
                            name_postfix = ''
                            # np.save(f"{ROOTPATH}/obs{name_postfix}.npy", obs.cpu().detach().numpy())
                            # np.save(f"{ROOTPATH}/syn{name_postfix}.npy", syn.cpu().detach().numpy())
                            loss = criterions(syn, obs)
                            # adj = torch.autograd.grad(loss, syn, create_graph=True)[0]
                            # np.save(f"{ROOTPATH}/adj.npy", adj.detach().cpu().numpy())
                            """HvP Start"""
                            # Perform a backward pass to compute the gradients
                            # grads = torch.autograd.grad(loss, [model.cell.geom.vp], create_graph=True)
                            # # Define a vector v with the same size as the model parameters
                            # v = [torch.randn_like(param) for param in [model.cell.geom.vp]]
                            # # Perform a forward pass with the vector v
                            # grads_v = torch.autograd.grad(grads, [model.cell.geom.vp], grad_outputs=v)
                            # # Perform a backward pass to compute the Hessian-vector product
                            # HvP = torch.autograd.grad(grads_v, [model.cell.geom.vp], retain_graph=True)
                            # np.save(os.path.join(cfg["geom"]["inv_savePath"], f"HvPE{epoch}S{shot_num}.npy"), HvP)
                            """HvP End"""
                            """START"""
                            # Model regularization
                            # l1_reg = 0
                            # for mname in model.cell.geom.model_parameters:
                            #     if mname == 'rho':
                            #         continue
                            #     l1_reg += torch.norm(model.cell.geom.__getattr__(mname), p=1)
                            # # Assign the weight of the model regulazation to %10 of the obj.
                            # alpha = loss.item()*1e-16
                            # loss += alpha*l1_reg
                            """END"""

                            loss.backward()

                            return loss.item()

                        # Run the closure
                        loss = closure()

                        GRAD = list()
                        for mname in model.cell.geom.pars_need_invert:
                            GRAD.append(model.cell.geom.__getattr__(mname).grad.cpu().detach().numpy())
                        GRAD = np.array(GRAD)
                        # Get the gram_schmidt_orthogonalization

                        # GRAD[1], GRAD[0] = gram_schmidt_orthogonalization(GRAD[1], GRAD[0])
                        comm.send((tasks, rank, GRAD, loss), dest=0, tag=1)

                comm.Barrier()

                """"Assigning and Saving"""
                if rank == 0:
                    pbar.close()
                    # Calculate the gradient of other ranks
                    grad2d[:] = np.sum(grad3d, axis=0)
                    np.save(f"{ROOTPATH}/loss.npy", loss)
                    np.save(f"{ROOTPATH}/grad3d.npy", grad3d)
                    # np.save(f"{ROOTPATH}/grad2d.npy", grad2d)
                    # Clean the grad3d
                    grad3d[:] = 0.
                # broadcast the gradient to other ranks
                comm.Bcast(grad2d, root=0)

                if rank!=0:
                    # Assign gradient of other ranks
                    for idx, para in enumerate(model.cell.geom.pars_need_invert):
                        var = model.cell.geom.__getattr__(para)
                        var.grad.data = to_tensor(grad2d[idx]).to(args.dev)

                    if args.grad_smooth:
                        model.cell.geom.gradient_smooth()

                    if args.grad_cut and isinstance(SEABED, torch.Tensor):
                        model.cell.geom.gradient_cut(SEABED, cfg['geom']['pml']['N'])                    # Gradient clip
                    #torch.nn.utils.clip_grad_norm_(model.cell.parameters(), 1e-2)
                    # Update the model parameters and learning rate
                    optimizers.step()
                    lr_scheduler.step()

                if rank==1:
                    # Save vel and grad
                    model.cell.geom.save_model(ROOTPATH, 
                                               paras=["vel", "grad"], 
                                               freq_idx=idx_freq,
                                               writer=writer,
                                               epoch=epoch)
