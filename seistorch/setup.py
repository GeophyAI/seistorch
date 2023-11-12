import os, tqdm
import importlib
from copy import deepcopy

import numpy as np
import torch

from seistorch.eqconfigure import Parameters
from seistorch.loss import Loss
from seistorch.io import SeisIO, SeisRecord
from seistorch.utils import read_pkl, ricker_wave, to_tensor
from seistorch.source import WaveSource
from seistorch.probe import WaveIntensityProbe

class SeisSetup:

    def __init__(self, cfg, args, logger):
        self.io = SeisIO(load_cfg=False)
        self.cfg = cfg
        self.args = args
        self.logger = logger

    def setup_batchsize(self):
        use_minibatch = self.cfg['training']['minibatch']

        num_shots_actual = self.setup_num_shots()

        if use_minibatch:
            batchsize = self.cfg['training']['batch_size']
        else:
            batchsize = min(num_shots_actual, self.args.num_batches)

        # How many tasks will be run on all GPU.
        num_batches = min(num_shots_actual, self.args.num_batches, batchsize)

        return batchsize, num_batches

    def setup_criteria(self, ):
        """Setup the loss functions for the model

        Args:
        Returns:
            torch.nn.module: The specified loss function.
        """
        ACOUSTIC = self.cfg['equation'] == 'acoustic'
        # The parameters needed to be inverted
        loss_names = set(self.args.loss.values())
        MULTI_LOSS = len(loss_names) > 1
        if not MULTI_LOSS or ACOUSTIC:
            self.logger.print("Only one loss function is used.")
            criterions = Loss(list(loss_names)[0]).loss(self.cfg)
        else:
            criterions = {k:Loss(v).loss(self.cfg) for k,v in self.args.loss.items()}
            self.logger.print(f"Multiple loss functions are used:\n {criterions}")
        return criterions

    def setup_device(self, rank):
        """Setup the device for the model

        Args:
            rank (int): The rank of the process.

        Returns:
            dev: The device for the model.
        """

        use_cuda = self.args.use_cuda and torch.cuda.is_available()

        if use_cuda:
            # Get the local_rank on each node
            torch.cuda.set_device(rank%torch.cuda.device_count())
            dev = torch.cuda.current_device()

        else:
            dev = torch.device('cpu')

        return dev

    def setup_file_system(self, ):
        self.logger.print("Setting up file system...")
        #seisrec = SeisRecord(self.cfg, logger=self.logger)
        seisrec = SeisRecord(self.cfg, logger=None)
        seisrec.setup(self.args.mode)
        return seisrec

    def setup_num_shots(self):
        # Read the source and receiver locations from the configuration file
        src_list, _ = self.io.read_geom(self.cfg)
        # Get the number of shots in the geometry file
        num_shots_in_geom = len(src_list)
        # Get the number of shots in the configure file
        num_shots_in_cfg = self.cfg['geom']['Nshots']
        # The number of shots used in the simulation
        num_shots_actual = min(num_shots_in_cfg, num_shots_in_geom)

        return num_shots_actual

    def setup_optimizer(self, 
                        model, 
                        idx_freq=0, 
                        implicit=False, 
                        grad_clamp=True,
                        *args, **kwargs):
        """Setup the optimizer for the model

        Args:
            model (RNN): The model to be optimized.
            cfg (dict): The configuration file.
        """
        lr = self.cfg['training']['lr']
        opt = self.cfg['training']['optimizer']
        epoch_decay = self.cfg['training']['lr_decay']
        scale_decay = self.cfg['training']['scale_decay']
        pars_need_by_eq = Parameters.valid_model_paras()[self.cfg['equation']]
        pars_need_invert = [k for k, v in self.cfg['geom']['invlist'].items() if v]

        # Setup the learning rate for each parameter
        paras_for_optim = []

        for para in pars_need_by_eq:
            # Set the learning rate for each parameter
            _lr = 0. if para not in pars_need_invert else lr[para]*scale_decay**idx_freq
            paras_for_optim.append({'params': model.cell.get_parameters(para, implicit=implicit), 
                                    'lr':_lr})
            eps = 1e-22 if not implicit else 1e-8

        opt_module = importlib.import_module('seistorch.optimizer')
        optimizers = getattr(opt_module, opt.capitalize())(paras_for_optim, eps=eps, grad_clamp=grad_clamp)

        # Setup the learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, epoch_decay, last_epoch=- 1, verbose=False)

        return optimizers, lr_scheduler

    def setup_pbar(self, stop, desc):

        return tqdm.trange(stop, position=0, desc=desc)

    def setup_seed(self,):
        # Set random seed
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

    def setup_seabed(self):
        use_seabed = True if 'seabed' in self.cfg['geom'].keys() else False
        if use_seabed:
            seabed = np.load(self.cfg['geom']['seabed'])
            SEABED = torch.from_numpy(seabed).to(self.args.dev)
        else:
            SEABED = None
        return SEABED

    def setup_tasks(self):

        # Bools
        use_minibatch = self.cfg['training']['minibatch']
        forward = self.args.mode == "forward"
        inversion = self.args.mode == "inversion"

        batchsize, num_batches = self.setup_batchsize()
        num_shots_actual = self.setup_num_shots()

        if forward:
            tasks = np.arange(num_shots_actual)

        if inversion:

            scales = len(self.cfg['geom']['multiscale'])
            epoch_per_scale = self.cfg['training']['N_epochs']
            num_iters = epoch_per_scale * scales
            shot_index = np.arange(num_shots_actual)

            if use_minibatch:
                tasks = iter([np.random.choice(shot_index, batchsize, replace=False) for _ in range(num_iters)])

            if not use_minibatch:
                tasks = iter([shot_index for _ in range(num_iters)])

        return tasks
            
    def setup_wavelet(self, ):
        """Setup the wavelet for the simulation.

        Returns:
            torch.Tensor: Tensor containing the wavelet.
        """
        if not self.cfg["geom"]["wavelet"]:
            self.logger.print("Using wavelet func.")
            x = ricker_wave(self.cfg['geom']['fm'], 
                            self.cfg['geom']['dt'], 
                            self.cfg['geom']['nt'], 
                            self.cfg['geom']['wavelet_delay'], 
                            inverse=self.cfg['geom']['wavelet_inverse'])
        else:
            self.logger.print("Loading wavelet from file")
            x = to_tensor(np.load(self.cfg["geom"]["wavelet"]))
        # Save the wavelet
        if 'ROOTPATH' in self.cfg.keys():
            self.logger.print(f"The wavelet is saved to {self.cfg['ROOTPATH']}.")
            np.save(os.path.join(self.cfg['ROOTPATH'], "wavelet.npy"), x.cpu().numpy())
        return x

    def update_pbar(self, pbar, freq_idx, local_epoch):

        num_scales = len(self.cfg['geom']['multiscale'])
        num_epochs = self.cfg['training']['N_epochs']

        pbar.set_description(f"Epoch {local_epoch+1}/{num_epochs} | Scale {freq_idx+1}/{num_scales}")

def setup_acquisition(shots, src_list, rec_list, cfg, *args, **kwargs):

    sources, receivers = [], []

    for shot in shots:
        src = setup_src_coords(src_list[shot], cfg['geom']['pml']['N'], cfg['geom']['multiple'])
        rec = setup_rec_coords(rec_list[shot], cfg['geom']['pml']['N'], cfg['geom']['multiple'])
        sources.append(src)
        receivers.extend(rec)

    return sources, receivers

def setup_device(rank, use_cuda=True):
    """Setup the device for the model

    Args:
        rank (int): The rank of the process.
        use_cuda (bool, optional): Whether use GPU or not. Defaults to True.

    Returns:
        dev: The device for the model.
    """

    use_cuda = use_cuda and torch.cuda.is_available()

    if use_cuda:
        # Get the local_rank on each node
        torch.cuda.set_device(rank%torch.cuda.device_count())
        dev = torch.cuda.current_device()

    else:
        dev = torch.device('cpu')

    return dev

def setup_device_by_rank(use_cuda=True, rank=0):
    """Setup the device for the model

    Args:
        use_cuda (bool, optional): Whether use GPU or not. Defaults to True.

    Returns:
        torch.device: The device for the model.
    """
    if use_cuda and torch.cuda.is_available():
        # Get the local_rank on each node
        torch.cuda.set_device(rank%torch.cuda.device_count())
        dev = torch.cuda.current_device()
    else:
        dev = torch.device('cpu')

    return dev

def setup_split_configs(cfg_path: str, chunk_size, mode, *args, **kwargs):
    IO = SeisIO(load_cfg=False)
    cfg = IO.read_cfg(cfg_path)
    # Get the path of the model
    key = 'initPath' if mode == 'inversion' else 'truePath'
    MODEL_PATH = cfg['geom'][key]
    # Get the parameters needed by the equation
    needed_model_paras = Parameters.valid_model_paras()[cfg['equation']]
    # Set the save path of the splitted model
    ROOT_PATH = os.path.dirname(MODEL_PATH['vp'])
    ROOT_PATH = os.path.join(ROOT_PATH, "chunk")
    os.makedirs(ROOT_PATH, exist_ok=True)
    new_config_paths = []
    # Split the configure file
    cfg_chunks = [deepcopy(cfg) for _ in range(chunk_size)]

    for param in needed_model_paras:
        cfg_chunk = cfg.copy()
        # Load the model parameter
        path = MODEL_PATH[param]
        if path is not None and os.path.exists(path):
            para_value = IO.read_vel(path, pmln=0)
        # Split the model parameter
        para_chunks = split_model_to_chunks(para_value, chunk_size, cfg['modelparallel']['overlap'], to_gpu=False)

        for idx, para_chunk in enumerate(para_chunks):
            m_savepath = f"{ROOT_PATH}/{param}_chunk_{idx}.npy"
            np.save(m_savepath, para_chunk)
            cfg_chunks[idx]['geom'][key][param] = m_savepath

    # Read the source and receiver locations from the configuration file
    srcs, recs = IO.read_geom(cfg)
    # Split the source and receiver locations to chunks
    new_srcs, new_recs = split_geom_to_chunks(srcs, recs, chunk_size, cfg['modelparallel']['overlap'], para_value.shape)

    for ichunk in range(chunk_size):

        src_savepath = f"{ROOT_PATH}/sources_chunk{ichunk}.pkl"
        rec_savepath = f"{ROOT_PATH}/receivers_chunk{ichunk}.pkl"

        cfg_chunks[ichunk]['geom']['sources'] = src_savepath
        cfg_chunks[ichunk]['geom']['receivers'] = rec_savepath
        
        IO.write_pkl(src_savepath, new_srcs[ichunk])
        IO.write_pkl(rec_savepath, new_recs[ichunk])

    # Write the splitted configure file
    for ichunk in range(chunk_size):
        config_savepath = f"{ROOT_PATH}/{mode}_chunk{ichunk}.yml"
        IO.write_cfg(config_savepath, cfg_chunks[ichunk])
        new_config_paths.append(config_savepath)

    return new_config_paths

def setup_rec_coords(coords, Npml, multiple=False):
    """Setup receiver coordinates.

    Args:
        coords (list): A list of coordinates.
        Npml (int): The number of PML layers.
        multiple (bool, optional): Whether use top PML or not. Defaults to False.

    Returns:
        WaveProbe: A torch.nn.Module receiver object.
    """

    # Coordinate are specified
    keys = ['x', 'y', 'z']
    kwargs = dict()

    # Without multiple
    for key, value in zip(keys, coords):
        kwargs[key] = [v + Npml if v is not None else None for v in value]

    # 2D case with multiple
    if 'z' not in kwargs.keys() and multiple:
        kwargs['y'] = [v - Npml if v is not None else None for v in kwargs['y']]

    # 3D case with multiple
    if 'z' in kwargs.keys() and multiple:
        raise NotImplementedError("Multiples in 3D case is not implemented yet.")
        #kwargs['z'] = [v-Npml for v in kwargs['z']]

    return [WaveIntensityProbe(**kwargs)]

def setup_src_rec(cfg: dict):
    """Read the source and receiver locations from the configuration file.

    Args:
        cfg (dict): The configuration file.

    Returns:
        tuple: Tuple containing: (source locations, 
        receiver locations of each shot, 
        full receiver locations,
        whether the receiver locations are fixed)
    """
    # Read the source and receiver locations from the configuration file
    assert os.path.exists(cfg["geom"]["sources"]), "Cannot found source file."
    assert os.path.exists(cfg["geom"]["receivers"]), "Cannot found receiver file."
    src_list = read_pkl(cfg["geom"]["sources"])
    rec_list = read_pkl(cfg["geom"]["receivers"])
    assert len(src_list)==len(rec_list), \
        "The lenght of sources and recev_locs must be equal."
    # Check whether the receiver locations are fixed
    fixed_receivers = all(rec_list[i]==rec_list[i+1] for i in range(len(rec_list)-1))
    # If the receiver locations are not fixed, use the model grids as the full receiver locations
    if not fixed_receivers: 
        print(f"Inconsistent receiver location detected.")
        receiver_counts = cfg['geom']['_oriNx']
        rec_depth = rec_list[0][1][0]
        full_rec_list = [[i for i in range(receiver_counts)], [rec_depth]*receiver_counts]
        # TODO: Add a warning here
        # The full receiver list should be the available receivers in rec_list.
    else:
        print(f"Receiver locations are fixed.")
        full_rec_list = rec_list[0]

    return src_list, rec_list, full_rec_list, fixed_receivers

def setup_src_coords(coords, Npml, multiple=False):
    """Setup source coordinates.

    Args:
        coords (list): A list of coordinates.
        Npml (int): The number of PML layers.
        multiple (bool, optional): Whether use top PML or not. Defaults to False.

    Returns:
        WaveSource: A torch.nn.Module source object.
    """
    # Coordinate are specified
    keys = ['x', 'y', 'z']
    kwargs = dict()
    # Padding the source location with PML
    for key, value in zip(keys, coords):
        if isinstance(value, (int, float)):
            kwargs[key] = value+Npml
        else:
            kwargs[key] = value # value = None

    # 2D case with multiple
    if 'z' not in kwargs.keys() and multiple and bool(kwargs['y']):
        kwargs['y'] -= Npml

    # 3D case with multiple
    if 'z' in kwargs.keys() and multiple:
        raise NotImplementedError("Multiples in 3D case is not implemented yet.")
        # kwargs['z'] -= Npml

    return WaveSource(**kwargs)

def split_geom_to_chunks(srcs, recs, chunk_num, overlap, shape):

    nz, nx = shape
    chunk_size = nx // chunk_num

    last_start = 0
    chunk_ranges = []
    # Get the range of each chunk in the whole model
    for i in range(chunk_num):
        start = last_start
        end = min(start+chunk_size, nx) if i != chunk_num-1 else nx
        record = (start, end)# if i == 0 else (start+overlap, end)
        chunk_ranges.append(record)
        last_start = end - overlap

    # The length of the src/rec is the shot number
    new_srcs = [[] for _ in range(chunk_num)]

    for src in srcs:
        src_x = src[0]
        src_z = src[1]
        for ichunk, range_ in enumerate(chunk_ranges):
            if range_[0] <= src_x < range_[1]:
                src_x = src_x if i==0 else src_x - range_[0]
                new_srcs[ichunk].append([src_x, src_z])
                #break
            else:
                new_srcs[ichunk].append([None, None])

    new_recs = [[ [[], []] for _ in range(len(recs))] for _ in range(chunk_num)]

    for ishot, rec in enumerate(recs):
        for rec_x, rec_z in zip(rec[0], rec[1]):

            ichunk = find_range_index(chunk_ranges, rec_x)

            new_rec_x = rec_x# if ichunk==0 else rec_x - range_[0]
            new_recs[ichunk][ishot][0].append(new_rec_x)
            new_recs[ichunk][ishot][1].append(rec_z)

    return new_srcs, new_recs

def split_model_to_chunks(model, chunk_num, overlap, to_gpu=True):
    """Split the model to chunks.

    Args:
        model (np.ndarray): Numpy array containing the model.
        chunk_num (int): The number of chunks.
        overlap (int): The overlap between chunks.
        to_gpu (bool, optional): Transfer to gpu. Defaults to True.

    Returns:
        list: List containing the splitted model.
    """

    nz, nx = model.shape
    chunk_size = nx // chunk_num
    chunks = []
    last_start = 0
    for i in range(chunk_num):
        start = last_start
        end = min(start+chunk_size, nx) if i != chunk_num-1 else nx
        chunk = model[:, start:end]
        if to_gpu:
            chunks.append(torch.from_numpy(chunk).to(f"cuda:{i}"))
        else:
            chunks.append(chunk)
        last_start = end - overlap

    return chunks

def find_range_index(range_list, num):
    for index, (start, end) in enumerate(range_list):
        if start <= num <= end:
            return index
    return None