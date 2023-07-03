"""Perform full waveform inversion."""
import argparse
import os
import time
import gc
import numpy as np
import setproctitle
import torch
import tqdm
import logging
# from skopt import Optimizer
from yaml import dump, load
from torch.utils.tensorboard import SummaryWriter
from ot.utils import proj_simplex

import wavetorch
from wavetorch.loss import Loss
# from tensorflow.keras.models import load_model
from wavetorch.model import build_model
from wavetorch.optimizer import NonlinearConjugateGradient as NCG
from wavetorch.optimizer import SophiaG
from wavetorch.eqconfigure import Shape, Parameters
from wavetorch.utils import cpu_fft, ricker_wave, roll, to_tensor, get_src_and_rec, DictAction, dict2table
from wavetorch.setup_source_probe import setup_src_coords, setup_rec_coords
from wavetorch.regularization import LaplacianRegularization
# from torchviz import make_dot
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('--num_threads', type=int, default=2,
                    help='Number of threads to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--encoding', action='store_true',
                    help='Use phase encoding to accelerate performance')
parser.add_argument('--gpuid', type=int, default=0,
                    help='which gpu is used for calculation')
parser.add_argument('--checkpoint', type=str,
                    help='checkpoint path for resuming training')
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
parser.add_argument('--opt', choices=['adam', 'lbfgs', 'ncg'], default='adam',
                    help='optimizer (adam)')
parser.add_argument('--loss', action=DictAction, nargs="+",
                    help='loss dictionary')
parser.add_argument('--save-path', default='',
                    help='the root path for saving results')
parser.add_argument('--lr', action=DictAction, nargs="+",
                    help='learning rate')
parser.add_argument('--global-lr', type=int, default=-1,
                    help='learning rate')
parser.add_argument('--batchsize', type=int, default=-1,
                    help='learning rate')
parser.add_argument('--grad-smooth', action='store_true',
                    help='Smooth the gradient or not')
parser.add_argument('--grad-cut', action='store_true',
                    help='Cut the boundaries of gradient or not')
parser.add_argument('--mode', choices=['inversion'], default='inversion',
                    help='forward modeling, inversion or reverse time migration mode')

if __name__ == '__main__':
    args = parser.parse_args()

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
    cfg, model = build_model(args.config, device=args.dev, mode=args.mode)

    # Set random seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    #model = torch.compile(model)

    # Set the name of the process
    setproctitle.setproctitle("coding_fwi")

    """Short cuts of the configures"""
    ELASTIC = cfg['equation'] in ['elastic', 'aec']
    ACOUSTIC = cfg['equation'] == 'acoustic'
    EPOCHS = cfg['training']['N_epochs']
    NSHOTS = cfg['geom']['Nshots']
    FILTER_ORDER = cfg['training']['filter_ord']
    MINIBATCH = cfg['training']['minibatch']
    BATCHSIZE = cfg['training']['batch_size'] if args.batchsize < 0 else args.batchsize
    PARS_NEED_INVERT = [k for k, v in cfg['geom']['invlist'].items() if v]
    PARS_NEED_BY_EQ = Parameters.valid_model_paras()[cfg['equation']]
    LR_DECAY = cfg['training']['lr_decay']
    SCALE_DECAY = cfg['training']['scale_decay']
    ROOTPATH = args.save_path if args.save_path else cfg["geom"]["inv_savePath"]
    GRAD_SMOOTH = args.grad_smooth
    GRAD_CUT = args.grad_cut
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
    src_list, rec_list = get_src_and_rec(cfg)

    model.to(args.dev)
    model.train()
    # In coding fwi, the probes are set only once.
    probes = setup_rec_coords(rec_list[0], cfg['geom']['pml']['N'])
    model.reset_probes(probes)
    # print(model.state_dict())
    # Resume training from checkpoint
    # assert os.path.exists(args.checkpoint), "Checkpoint not found"
    # Load the checkpoint
    # checkpoint = torch.load(args.checkpoint)

    # print("model state dict:", model.state_dict())
    # print("\n\n\n\n\n\n")
    # print(checkpoint)
    # # model.load_state_dict(checkpoint['model_state_dict'])
    # exit()

    # print(probes)
    """# Read the wavelet"""
    if not cfg["geom"]["wavelet"]:
        print("Using wavelet func.")
        x = ricker_wave(cfg['geom']['fm'], cfg['geom']['dt'], cfg['geom']['nt'])
    else:
        print("Loading wavelet from file")
        x = to_tensor(np.load(cfg["geom"]["wavelet"]))
    shape = Shape(cfg)

    """---------------------------------------------"""
    """-------------------INVERSION-----------------"""
    """---------------------------------------------"""

    """Write configure file to the inversion folder"""
    cfg["loss"] = args.loss

    with open(os.path.join(ROOTPATH, "configure.yml"), "w") as f:
        dump(cfg, f)
    # convert the configure to table
    cfg_table = dict2table(cfg)
    for _t in cfg_table:
        logging.info(cfg_table)
    """Define the misfit function for different parameters"""
    # The parameters needed to be inverted
    loss_names = set(args.loss.values())
    MULTI_LOSS = len(loss_names) > 1
    if not MULTI_LOSS or ACOUSTIC:
        print("Only one loss function is used.")
        criterions = Loss(list(loss_names)[0]).loss()
    else:
        criterions = {k:Loss(v).loss() for k,v in args.loss.items()}
        print(f"Multiple loss functions are used:\n {criterions}")

    """Only rank0 will read the full band data"""
    """Rank0 will broadcast the data after filtering"""
    full_band_data = np.load(cfg['geom']['obsPath'])
    filtered_data = np.zeros(shape.record3d, dtype=np.float32)
    coding_obs = torch.zeros(shape.record2d, device=args.dev)
    coding_wavelet = torch.zeros((BATCHSIZE, shape.nt), device=args.dev)
    loss = np.zeros((len(cfg['geom']['multiscale']), EPOCHS), np.float32)

    """Loop over all scale"""
    for idx_freq, freq in enumerate(cfg['geom']['multiscale']):
        # Restart the optimizer for each scale
        # Old codes
        if args.opt=='adam':
            paras_for_optim = []
            for para in PARS_NEED_BY_EQ:
                # Set the learning rate for each parameter
                _lr = 0. if para not in PARS_NEED_INVERT else args.lr[para]*SCALE_DECAY**idx_freq
                paras_for_optim.append({'params': model.cell.get_parameters(para), 
                                        'lr':_lr})
            optimizers = torch.optim.Adam(paras_for_optim, betas=(0.9, 0.999), eps=1e-22)

            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, LR_DECAY, last_epoch=- 1, verbose=False)


        logging.info(f"Info. of optimizers:{optimizers}")
        logging.info(f"Data filtering: frequency:{freq}")
        # Filter both record and ricker
        filtered_data[:] = cpu_fft(full_band_data.copy(), cfg['geom']['dt'], N=FILTER_ORDER, low=freq, axis = 1, mode='lowpass')
        # Low pass filtered wavelet
        if isinstance(x, torch.Tensor): x = x.numpy()
        lp_wavelet = cpu_fft(x.copy(), cfg['geom']['dt'], N=FILTER_ORDER, low=freq, axis=0, mode='lowpass')
        lp_wavelet = torch.unsqueeze(torch.from_numpy(lp_wavelet), 0)
        pbar = tqdm.trange(EPOCHS)
        """Loop over all epoches"""
        for epoch in range(EPOCHS):

            # Zero the obs tensor
            coding_obs.zero_()
            coding_wavelet.zero_()
            pbar.set_description(f"F{idx_freq}E{epoch}")
            if MINIBATCH:
                shots = np.random.choice(np.arange(NSHOTS), BATCHSIZE, replace=False)
            else:
                shots = np.arange(NSHOTS)
            logging.info(f"Encoding shots: {shots}")
            sources = []
            receivers = []
            # Get the coding shot numbers and coding data
            for i, shot in enumerate(shots):
                src = setup_src_coords(src_list[shot], cfg['geom']['pml']['N'])
                sources.append(src)
                wave_temp, d_temp = roll(lp_wavelet, filtered_data[shot])
                coding_wavelet[i] = to_tensor(wave_temp).to(args.dev)
                coding_obs += to_tensor(d_temp).to(args.dev)
            
            """Calculate encoding gradient"""
            def closure():
                optimizers.zero_grad()
                # for opt in optimizers.values():
                #     opt.zero_grad()
                # Get the super shot gathersh
                model.reset_sources(sources)
                # model.cell.geom.reset_random_boundary()
                ypred = model(coding_wavelet)
                # loss = criterion(ypred, coding_obs, model.cell.geom.vp)
                # np.save(f"{ROOTPATH}/syn.npy", ypred.cpu().detach().numpy())
                # np.save(f"{ROOTPATH}/obs.npy", coding_obs.cpu().detach().numpy())
                if not MULTI_LOSS:
                    # One loss function for all parameters
                    loss = criterions(ypred, coding_obs)
                    loss.backward()
                if MULTI_LOSS:
                    # Different loss function for different parameters
                    LOSSES = []
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
                        loss = criterion(ypred, coding_obs)
                        # if the para is the last loss, do not retain the graph
                        retain_graph = False if _i == len(criterions)-1 else True
                        loss.backward(retain_graph=retain_graph)

                    # Reset the requires_grad of all parameters to True,
                    # because we need to save the gradient of all parameters in <save_model>
                    for p in PARS_NEED_INVERT:
                        model.cell.geom.__getattr__(p).requires_grad = True

                # adjoint = torch.autograd.grad(loss, ypred)[0]
                # np.save(f"{ROOTPATH}/adjoint_{cfg['loss']}.npy", 
                #         adjoint.detach().cpu().numpy())
                # model.cell.geom.reset_random_boundary()
                return loss

            # Run the closure
            if args.opt == "ncg":
                loss[idx_freq][epoch] = optimizers.step(closure).item()
            else:
                # loss = closure()
                loss[idx_freq][epoch] = closure().item()

                if GRAD_SMOOTH:
                    model.cell.geom.gradient_smooth(sigma=2)
                if GRAD_CUT:
                    model.cell.geom.gradient_cut()
                # Update the parameters with different optimizer instance
                # for para in PARS_NEED_INVERT:
                #     optimizers[para].step()
                optimizers.step()
                lr_scheduler.step()
                vp = model.cell.geom.vp.data.detach().cpu().numpy()
                vp[50:50+48,:] = 1500.
                model.cell.geom.vp.data = torch.from_numpy(vp).to(torch.cuda.current_device())

                # model.cell.geom.vp.data = proj_simplex(model.cell.geom.vp)

            # Saving checkpoint
            torch.save({'epoch': idx_freq*EPOCHS+epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizers.state_dict()}, 
                        f"{ROOTPATH}/ckpt_{epoch}.pt")

            logging.info(f"Freq {idx_freq:02d} Epoch {epoch:02d} loss: {loss[idx_freq][epoch]}")
            writer.add_scalar(f"Loss", loss[idx_freq][epoch], global_step=idx_freq*EPOCHS+epoch)
            # if args.opt!="ncg":
            #    lr_scheduler.step()
            pbar.update(1)
            # Save vel and grad
            np.save(os.path.join(ROOTPATH, "loss.npy"), loss)
            writer.add_scalar(f"GPU allocated", torch.cuda.max_memory_allocated()/ (1024 ** 3), global_step=idx_freq*EPOCHS+epoch)
            writer.add_scalar(f"CPU Usage", torch.cuda.memory_stats()['allocated_bytes.all.current']/ (1024 ** 3), global_step=idx_freq*EPOCHS+epoch)
            model.cell.geom.save_model(ROOTPATH, 
                                       paras=["vel", "grad"], 
                                       freq_idx=idx_freq, 
                                       epoch=epoch,
                                       writer=writer, max_epoch=EPOCHS)

    # Close the summary writer    
    writer.close()
            