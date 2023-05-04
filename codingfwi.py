"""Perform full waveform inversion."""
import setproctitle
import wavetorch
import numpy as np
from functools import partial
import argparse, os, time, tqdm, torch
from wavetorch.utils import ricker_wave, to_tensor, cpu_fft, roll
# from tensorflow.keras.models import load_model
from wavetorch.model import build_model
from wavetorch.loss import Loss
from wavetorch.optimizer import gram_schmidt_orthogonalization
from wavetorch.optimizer import NonlinearConjugateGradient as NCG
from wavetorch.shape import Shape
# from skopt import Optimizer
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from wavetorch.setup_source_probe import setup_src_coords_customer, get_sources_coordinate_list

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
                    help='loss')
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
    ELASTIC = cfg['equation'] == 'elastic'
    ACOUSTIC = cfg['equation'] == 'acoustic'
    EPOCHS = cfg['training']['N_epochs']
    NSHOTS = cfg['geom']['Nshots']
    LEARNING_RATE = cfg['training']['lr']
    NORMALIZATION = cfg["training"]["normalize"]
    FILTER_ORDER = cfg['training']['filter_ord']
    MINIBATCH = cfg['training']['minibatch']
    BATCHSIZE = cfg['training']['batch_size']
    ROOTPATH = cfg["geom"]["inv_savePath"]
    # Check the working folder
    if not os.path.exists(ROOTPATH):
        os.makedirs(ROOTPATH, exist_ok=True)
    ### Get source-x and source-y coordinate in grid cells
    source_x_list, source_y_list = get_sources_coordinate_list(cfg)

    model.to(args.dev)
    model.train()

    """# Read the wavelet"""
    x = ricker_wave(cfg['geom']['fm'], cfg['geom']['dt'], cfg['geom']['nt'])

    shape = Shape(cfg)

    """---------------------------------------------"""
    """-------------------INVERSION-----------------"""
    """---------------------------------------------"""

    """Write configure file to the inversion folder"""
    with open(os.path.join(cfg['geom']['inv_savePath'], "configure.yml"), "w") as f:
        dump(cfg, f)

    """Define Optimizer"""
    if args.opt=='adam':
        if ACOUSTIC:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-16)
        if ELASTIC: 
            optimizer = torch.optim.Adam([
                    {'params': model.cell.get_parameters('vp'), 'lr':LEARNING_RATE},
                    {'params': model.cell.get_parameters('vs'), 'lr':LEARNING_RATE/1.73},
                    {'params': model.cell.get_parameters('rho'), 'lr':0.}], 
                    betas=(0.9, 0.999), eps=1e-16)
            
    if args.opt == "ncg":
        optimizer = NCG(model.parameters(), lr=10., max_iter_line_search=10)

            # opt_bayesian = Optimizer(dimensions=[(-10.0, 10.0)]*shape.numel, 
            #                             base_estimator="gp", n_initial_points=0, acq_func="EI")


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
    # The gradient in rank0 is a 3D array.
    grad2d = np.zeros(shape.grad2d, np.float32)

    """Loop over all scale"""
    for idx_freq, freq in enumerate(cfg['geom']['multiscale']):

        print(f"Data filtering: frequency:{freq}")
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
            sources = []

            # Get the coding shot numbers and coding data
            for i, shot in enumerate(shots):
                src = setup_src_coords_customer(source_x_list[shot],
                                                source_y_list[shot],
                                                cfg['geom']['Nx'],
                                                cfg['geom']['Ny'],
                                                cfg['geom']['pml']['N'])
                sources.append(src)
                wave_temp, d_temp = roll(lp_wavelet, filtered_data[shot])
                coding_wavelet[i] = to_tensor(wave_temp).to(args.dev)
                coding_obs += to_tensor(d_temp).to(args.dev)

            """Calculate one shotye gradient"""
            def closure():
                optimizer.zero_grad()
                # Get the super shot gather
                model.reset_sources(sources)
                #ypred = model(lp_wavelet)
                ypred = model(coding_wavelet)
                #np.save("/mnt/others/DATA/Inversion/RNN/coding_visco/ypred.npy", ypred.cpu().detach().numpy())

                loss = criterion(ypred, coding_obs)
                loss.backward()

                return loss#.item()

            # Run the closure
            # loss[idx_freq][epoch] = closure(sources)
            if args.opt == "ncg":
                loss[idx_freq][epoch] = optimizer.step(closure).item()
            else:
                loss[idx_freq][epoch] = closure().item()
                optimizer.step()

            if args.opt!="ncg":
                lr_scheduler.step()
            pbar.update(1)
            # Save vel and grad
            np.save(os.path.join(cfg['geom']['inv_savePath'], "loss.npy"), loss)
            model.cell.geom.save_model(cfg['geom']['inv_savePath'], 
                                        paras=["vel", "grad"], 
                                        freq_idx=idx_freq, 
                                        epoch=epoch)