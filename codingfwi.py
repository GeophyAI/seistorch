"""Perform full waveform inversion."""
import argparse
import logging
import os
import time
import setproctitle
import torch
import tqdm

#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# logging.basicConfig(level=logging.ERROR)
import torch._dynamo.config as tdf
import torch._inductor.config as tif
tdf.log_level = logging.ERROR
tif.debug = False

import numpy as np
import setproctitle
from torch.utils.tensorboard import SummaryWriter
from yaml import dump

from seistorch.eqconfigure import Shape
# from tensorflow.keras.models import load_model
from seistorch.model import build_model
from seistorch.setup import *
from seistorch.utils import (DictAction, cpu_fft, dict2table,
                             low_pass, roll, roll2, to_tensor)

# from torchviz import make_dot
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('--num_threads', type=int, default=2,
                    help='Number of threads to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--gpuid', type=int, default=0,
                    help='which gpu is used for calculation')
parser.add_argument('--checkpoint', type=str,
                    help='checkpoint path for resuming training')
parser.add_argument('--opt', choices=['adam', 'lbfgs', 'ncg'], default='adam',
                    help='optimizer (adam)')
parser.add_argument('--loss', action=DictAction, nargs="+",
                    help='loss dictionary')
parser.add_argument('--save-path', default='',
                    help='the root path for saving results')
parser.add_argument('--lr', action=DictAction, nargs="+",
                    help='learning rate')
parser.add_argument('--batchsize', type=int, default=-1,
                    help='learning rate')
parser.add_argument('--grad-smooth', action='store_true',
                    help='Smooth the gradient or not')
parser.add_argument('--grad-cut', action='store_true',
                    help='Cut the boundaries of gradient or not')
parser.add_argument('--mode', choices=['inversion'], default='inversion',
                    help='forward modeling, inversion or reverse time migration mode')
parser.add_argument('--source-encoding', action='store_true', default=True,
                    help='PLEASE DO NOT CHANGE THE DEFAULT VALUE.')

if __name__ == '__main__':

    args = parser.parse_args()
    # Setup the device
    if args.use_cuda and torch.cuda.is_available():
        # Get the local_rank on each node
        torch.cuda.set_device(args.gpuid)
        args.dev = torch.cuda.current_device()
        print("Configuration: %s" % args.config)
        print("Using CUDA for calculation")
        print(f"Using {args.opt}")
    else:
        args.dev = torch.device('cpu')
        print("Configuration: %s" % args.config)
        print("Using CPU for calculation")

    'Sets the number of threads used for intraop parallelism on CPU.'
    torch.set_num_threads(args.num_threads)
    # Build model
    cfg, model = build_model(args.config, device=args.dev, mode=args.mode, source_encoding=args.source_encoding)
    #exit()
    # Set random seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Set the name of the process
    setproctitle.setproctitle("coding_fwi")

    """Short cuts of the configures"""
    ELASTIC = cfg['equation'] in ['elastic', 'aec']
    ACOUSTIC = cfg['equation'] == 'acoustic'
    EPOCHS = cfg['training']['N_epochs']
    NSHOTS = cfg['geom']['Nshots']
    FILTER_ORDER = cfg['training']['filter_ord']
    IMPLICIT = cfg['training']['implicit']['use']
    MINIBATCH = cfg['training']['minibatch']
    BATCHSIZE = cfg['training']['batch_size'] if args.batchsize < 0 else args.batchsize
    PARS_NEED_INVERT = [k for k, v in cfg['geom']['invlist'].items() if v]
    ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]
    SEABED = np.load(cfg['geom']['seabed']) if 'seabed' in cfg['geom'].keys() else None
    SEABED = torch.from_numpy(SEABED).to(args.dev) if SEABED is not None else None
    # Check the working folder
    if not os.path.exists(ROOTPATH):
        os.makedirs(ROOTPATH, exist_ok=True)
    # Configure the logger
    logging.basicConfig(level=logging.DEBUG,  # Set the log level to DEBUG (the lowest level)
                        format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log format
                        filename=f'{ROOTPATH}/log.log',  # Specify the log file name
                        filemode='w')  # Set the file mode to write mode
    writer = SummaryWriter(os.path.join(ROOTPATH, "logs"))
    print(f"The results will be saving at '{ROOTPATH}'")
    print(f"BATCHSIZE: {args.batchsize}")
    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list, full_rec_list, fixed_receivers = setup_src_rec(cfg)
    #model = torch.compile(model) #, mode="max-autotune"
    # Send the model to the device(CPU/GPU)
    model.to(args.dev)
    model.train()
    #model = torch.compile(model)
    # In coding fwi, the probes are set only once, 
    # because they are fixed with respect to moving source.
    probes = setup_rec_coords(full_rec_list, cfg['geom']['pml']['N'], cfg['geom']['multiple'])
    model.reset_probes(probes)

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
    """Write configure file to the inversion folder"""
    cfg["loss"] = args.loss
    cfg["ROOTPATH"] = ROOTPATH
    cfg['training']['lr'] = args.lr
    cfg['training']['batchsize'] = BATCHSIZE
    with open(os.path.join(ROOTPATH, "configure.yml"), "w") as f:
        dump(cfg, f)
    # print(probes)
    """# Read the wavelet"""
    x = setup_wavelet(cfg)

    shape = Shape(cfg)

    """---------------------------------------------"""
    """-------------------INVERSION-----------------"""
    """---------------------------------------------"""

    # convert the configure to table
    cfg_table = dict2table(cfg)
    for _t in cfg_table: logging.info(cfg_table)
    """Define the misfit function for different parameters"""
    # The parameters needed to be inverted
    criterions = setup_criteria(cfg, args.loss)
    MULTI_LOSS = isinstance(criterions, dict)

    """Only rank0 will read the full band data"""
    """Rank0 will broadcast the data after filtering"""
    full_band_data = np.load(cfg['geom']['obsPath'], allow_pickle=True)
    NSHOTS = min(NSHOTS, full_band_data.shape[0])
    #lp_rec = np.zeros(shape.record3d, dtype=np.float32)
    #coding_obs = torch.zeros(shape.record2d, device=args.dev)
    coding_obs = torch.zeros_like(to_tensor(full_band_data[0]), device=args.dev)
    coding_wav = torch.zeros((BATCHSIZE, shape.nt), device=args.dev)
    loss = np.zeros((len(cfg['geom']['multiscale']), EPOCHS), np.float32)
    #arrival_mask = np.load(cfg['geom']['arrival_mask'], allow_pickle=True)
    # The total number of epochs is the number of epochs times the number of scales
    for epoch in range(EPOCHS*len(cfg['geom']['multiscale'])):

        model.cell.geom.step()
        # Reset for each scale
        idx_freq, local_epoch = divmod(epoch, EPOCHS)
        if local_epoch==0:
            freq = cfg['geom']['multiscale'][idx_freq]
            # reset the optimizer
            optimizers, lr_scheduler = setup_optimizer(model, cfg, idx_freq, IMPLICIT)
            # filter the data
            # Filter both record and ricker
            lp_rec = low_pass(full_band_data.copy(), cfg['geom']['dt'], N=FILTER_ORDER, low=freq, axis=0, threads=args.num_threads)
            # Low pass filtered wavelet
            if isinstance(x, torch.Tensor): x = x.numpy()
            lp_wav = cpu_fft(x.copy(), cfg['geom']['dt'], N=FILTER_ORDER, low=freq, axis=0, mode='lowpass')
            lp_wav = torch.unsqueeze(torch.from_numpy(lp_wav), 0)

            logging.info(f"Info. of optimizers:{optimizers}")
            logging.info(f"Data filtering: frequency:{freq}")
            pbar = tqdm.trange(EPOCHS)

        # Clear the coding tensor
        coding_obs.zero_()
        coding_wav.zero_()
        pbar.set_description(f"F{idx_freq}E{local_epoch}")
        shots = np.random.choice(np.arange(NSHOTS), BATCHSIZE, replace=False) if MINIBATCH else np.arange(NSHOTS)
        sources, receivers = [], []

        # Get the coding shot numbers and coding data
        for i, shot in enumerate(shots):
            #shot = 335
            src = setup_src_coords(src_list[shot], cfg['geom']['pml']['N'], cfg['geom']['multiple'])
            sources.append(src)
            # For OBN data, the receivers are fixed and the receivers are the same for all shots
            # so we only need to setup the receivers once, and the data can be summed immediately
            # But for non-fixed receivers, we need to setup the receivers for each shot,
            # reconstruct a pseudo-OBN data and then sum them up
            
            # wave_temp, d_temp = roll(lp_wav, lp_rec[shot] * arrival_mask[shot])
            wave_temp, d_temp = roll(lp_wav, lp_rec[shot])
            coding_wav[i] = to_tensor(wave_temp).to(args.dev)
            if fixed_receivers:
                coding_obs += to_tensor(d_temp).to(args.dev)
            else:
                index = [int(x) for x in rec_list[shot][0]]
                coding_obs[..., index,:] += to_tensor(d_temp).to(args.dev)


        """Calculate encoding gradient"""
        def closure():
            optimizers.zero_grad()
            # Reset sources of super shot gathers
            model.reset_sources(sources)
            coding_syn = model(coding_wav)
            # loss = criterion(coding_syn, coding_obs, model.cell.geom.vp)
            np.save(f"{ROOTPATH}/syn.npy", coding_syn.cpu().detach().numpy())
            np.save(f"{ROOTPATH}/obs.npy", coding_obs.cpu().detach().numpy())
            if not MULTI_LOSS:
                # One loss function for all parameters
                loss = criterions(coding_syn, coding_obs.unsqueeze(0))
                # adj = torch.autograd.grad(loss, coding_syn)[0]
                # np.save(f"{ROOTPATH}/adj.npy", adj.detach().cpu().numpy())
                loss.backward() #retain_graph=True
                # TODO: HESSIAN
                #loss.backward(create_graph=True)
                # grad = model.cell.geom.vp.grad.detach().clone()
                # model.cell.geom.vp.grad.data.zero_()
                # model.cell.geom.vp.grad.backward(torch.ones_like(grad))
                # np.save(f"{ROOTPATH}/grad.npy", grad.cpu().detach().numpy())
                # np.save(f"{ROOTPATH}/hess.npy", model.cell.geom.vp.grad.cpu().detach().numpy())

            if MULTI_LOSS:
                # Different loss function for different parameters
                for _i, (para, criterion)in enumerate(criterions.items()):
                    # set the requires_grad of other parameters to False
                    for p in PARS_NEED_INVERT:
                        # if the parameter is not the one to be inverted,
                        # set the requires_grad to False
                        # if the parameter is the one to be inverted,
                        # set the requires_grad to True
                        requires_grad = False if p != para else True
                        model.cell.geom.__getattr__(p).requires_grad = requires_grad
                    # Calculate the loss
                    loss = criterion(coding_syn, coding_obs)
                    # adj = torch.autograd.grad(loss, coding_syn)[0]
                    # np.save(f"{ROOTPATH}/adj.npy", adj.detach().cpu().numpy())
                    # if the para is the last loss, do not retain the graph
                    retain_graph = False if _i == len(criterions)-1 else True
                    loss.backward(retain_graph=retain_graph)

                # Reset the requires_grad of all parameters to True,
                # because we need to save the gradient of all parameters in <save_model>
                for p in PARS_NEED_INVERT:
                    model.cell.geom.__getattr__(p).requires_grad = True

            return loss

        # Run the closure
        loss[idx_freq][local_epoch] = closure().item()

        if args.grad_smooth:
            model.cell.geom.gradient_smooth(sigma=1)

        if args.grad_cut and isinstance(SEABED, torch.Tensor):
            model.cell.geom.gradient_cut(SEABED, cfg['geom']['pml']['N'])

        # Update the parameters
        optimizers.step()
        lr_scheduler.step()

        # Saving checkpoint
        # torch.save({'epoch': local_epoch, 
        #             'model_state_dict': model.state_dict(), 
        #             'optimizer_state_dict': optimizers.state_dict()}, 
        #             f"{ROOTPATH}/ckpt_{local_epoch}.pt")

        pbar.update(1)
        # Save vel and grad

        model.cell.geom.save_model(ROOTPATH, 
                                   paras=["vel", "grad"], 
                                   freq_idx=idx_freq, 
                                   epoch=local_epoch,
                                   writer=writer, max_epoch=EPOCHS)

        # logging
        logging.info(f"Encoding shots: {shots}")
        logging.info(f"Freq {idx_freq:02d} Epoch {local_epoch:02d} loss: {loss[idx_freq][local_epoch]}")

        # Add scalars to tensorboard
        writer.add_scalar(f"Loss", loss[idx_freq][local_epoch], global_step=epoch)
        writer.add_scalar(f"GPU allocated", torch.cuda.max_memory_allocated()/ (1024 ** 3), global_step=epoch)
        writer.add_scalar(f"CPU Usage", torch.cuda.memory_stats()['allocated_bytes.all.current']/ (1024 ** 3), global_step=epoch)

        np.save(os.path.join(ROOTPATH, "loss.npy"), loss)

    writer.close()