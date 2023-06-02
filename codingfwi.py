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
from torchvision.utils import make_grid

import wavetorch
from wavetorch.loss import Loss
# from tensorflow.keras.models import load_model
from wavetorch.model import build_model
from wavetorch.optimizer import NonlinearConjugateGradient as NCG
from wavetorch.eqconfigure import Shape
from wavetorch.utils import cpu_fft, ricker_wave, roll, to_tensor, get_src_and_rec
from wavetorch.setup_source_probe import setup_src_coords, setup_rec_coords
from wavetorch.cell import WaveCell
from wavetorch.rnn import WaveRNN
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
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
parser.add_argument('--opt', choices=['adam', 'lbfgs', 'ncg'], default='adam',
                    help='optimizer (adam)')
parser.add_argument('--loss', default='mse',
                    help='loss function')
parser.add_argument('--save-path', default='',
                    help='the root path for saving results')
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
    LEARNING_RATE = cfg['training']['lr'] if args.global_lr < 0 else args.global_lr
    FILTER_ORDER = cfg['training']['filter_ord']
    MINIBATCH = cfg['training']['minibatch']
    BATCHSIZE = cfg['training']['batch_size'] if args.batchsize < 0 else args.batchsize
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
    print(f"LEARNING_RATE of VP: {LEARNING_RATE}")
    print(f"BATCHSIZE: {args.batchsize}")
    ### Get source-x and source-y coordinate in grid cells
    src_list, rec_list = get_src_and_rec(cfg)

    model.to(args.dev)
    model.train()
    # In coding fwi, the probes are set only once.
    probes = setup_rec_coords(rec_list[0], cfg['geom']['pml']['N'])
    model.reset_probes(probes)
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

    # loss_weights = torch.autograd.Variable(torch.ones(3), requires_grad=True)

    """Define Optimizer"""
    if args.opt=='adam':
        # optimizer = torch.optim.Adam([
        #         {'params': model.cell.get_parameters('vp'), 'lr':LEARNING_RATE}], 
        #         betas=(0.9, 0.999), eps=1e-20)
        # optimizer = torch.optim.Adam([
        #         {'params': model.cell.get_parameters('vp'), 'lr':LEARNING_RATE},
        #         {'params': model.cell.get_parameters('rho'), 'lr':LEARNING_RATE/1.73}], 
        #         betas=(0.9, 0.999), eps=1e-20)
        if ACOUSTIC:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-20)
        if ELASTIC: 
            optimizer = torch.optim.Adam([
                    {'params': model.cell.get_parameters('vp'), 'lr':LEARNING_RATE},
                    {'params': model.cell.get_parameters('vs'), 'lr':LEARNING_RATE/1.73},
                    {'params': model.cell.get_parameters('rho'), 'lr':0.}], 
                    betas=(0.9, 0.999), eps=1e-20)
            
    if args.opt == "ncg":
        optimizer = NCG(model.parameters(), lr=10., max_iter_line_search=10)

    """Define the learning rate decay"""
    if args.opt!="ncg":
        lr_milestones = [EPOCHS*(i+1) for i in range(len(cfg['geom']['multiscale']))]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=lr_milestones, 
                                                            gamma = cfg['training']['lr_decay'], 
                                                            verbose=False)

    """Define the misfit function"""
    criterion = Loss(args.loss).loss()

    """Only rank0 will read the full band data"""
    """Rank0 will broadcast the data after filtering"""
    full_band_data = np.load(cfg['geom']['obsPath'])
    filtered_data = np.zeros(shape.record3d, dtype=np.float32)
    coding_obs = torch.zeros(shape.record2d, device=args.dev)
    coding_wavelet = torch.zeros((BATCHSIZE, shape.nt), device=args.dev)
    loss = np.zeros((len(cfg['geom']['multiscale']), EPOCHS), np.float32)

    """Loop over all scale"""
    for idx_freq, freq in enumerate(cfg['geom']['multiscale']):

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
                optimizer.zero_grad()#set_to_none=True
                # Get the super shot gathersh
                # Reset model
                # cell = WaveCell(model.cell.geom, model.cell.forward_func, model.cell.backward_func)
                # # Build RNN
                # model = WaveRNN(cell)

                model.reset_sources(sources)
                ypred = model(coding_wavelet)
                loss = criterion(ypred, coding_obs)
                loss.backward()
                return loss

            # Run the closure
            if args.opt == "ncg":
                loss[idx_freq][epoch] = optimizer.step(closure).item()
            else:
                loss[idx_freq][epoch] = closure().item()

                if GRAD_SMOOTH:
                    model.cell.geom.gradient_smooth(sigma=2)
                if GRAD_CUT:
                    model.cell.geom.gradient_cut()

                optimizer.step()

            logging.info(f"Freq {idx_freq:02d} Epoch {epoch:02d} loss: {loss[idx_freq][epoch]}")
            writer.add_scalar(f"Loss", loss[idx_freq][epoch], global_step=idx_freq*EPOCHS+epoch)
            if args.opt!="ncg":
               lr_scheduler.step()
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
            