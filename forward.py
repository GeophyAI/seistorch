"""Perform full waveform inversion."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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
from seistorch.coords import single2batch2, offset_with_boundary
from seistorch.setup import *
from seistorch.utils import (DictAction, to_tensor, nestedlist2tensor)

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
parser.add_argument('--mode', choices=['forward'], default='forward',
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

                src, rec = offset_with_boundary(np.array(src_list)[shots], np.array(rec_list)[shots], cfg)
                
                src, rec = to_tensor(src), to_tensor(rec)

                batched_source, batched_probes = single2batch2(src, rec, cfg, args.dev) # padding, in batch
                # model.cell.geom.step() # random boundary
                y = model(x, None, batched_source, batched_probes)
                record = y.numpy()
            
            comm.send((tasks, rank, record), dest=0, tag=1)

    comm.Barrier()

    """Save the modeled data, which stored in rank0 <record>"""
    if MASTER:
        pbar.close()
        record.write()
        seislog.print("Modeling done, data will be writing to disk.")
