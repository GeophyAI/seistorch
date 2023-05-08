import os
import numpy as np
import torch, struct, socket
from scipy import signal
from typing import Any, Iterable, List, Tuple
import torch
import warnings
import weakref
from typing import Any, Iterable, List, Tuple

def to_tensor(x, dtype=None):
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if type(x) is np.ndarray:
        return torch.from_numpy(x).type(dtype=dtype)
    else:
        return torch.as_tensor(x, dtype=dtype)


def set_dtype(dtype=None):
    if dtype == 'float32' or dtype is None:
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError('Unsupported data type: %s; should be either float32 or float64' % dtype)


def window_data(X, window_length):
    """Window the sample, X, to a length of window_length centered at the middle of the original sample
    """
    return X[int(len(X) / 2 - window_length / 2):int(len(X) / 2 + window_length / 2)]

def accuracy_onehot(y_pred, y_label):
    """Compute the accuracy for a onehot
    """
    return (y_pred.argmax(dim=1) == y_label).float().mean().item()


def normalize_power(X):
    return X / torch.sum(X, dim=1, keepdim=True)
    
def ricker_wave(fm, dt, T, delay = 500, dtype='tensor'):
    """
        Ricker-like wave.
    """
    ricker = []
    delay = delay * dt 
    for i in range(T):
        c = np.pi * fm * (i * dt - delay) #  delay
        temp = (1-2*np.power(c, 2)) * np.exp(-np.power(c, 2))
        ricker.append(temp)
    if dtype == 'numpy':
        return np.array(ricker)
    else:
        return torch.from_numpy(np.array(ricker).astype(np.float32))

def cpu_fft(d, dt, N = 5, low = 5, if_plot = True, axis = -1, mode = 'lowpass'):
    """
        implementation of fft.
    """
    wn = 2*low/(1/dt)
    b, a = signal.butter(N, wn, mode)
    d_filter = signal.filtfilt(b, a, d, axis = axis)
    return d_filter.astype(np.float32)
    
def pad_by_value(d, pad, mode = 'double'):
    """pad the input by <pad>
    """
    if mode == 'double':
        return d + 2*pad
    else:
        return d + pad
        
def load_file_by_type(filepath, shape = None, pml_width = None):
    """load data files, differs by its type
    """
    fileType = filepath.split('/')[-1].split('.')[-1]
    if fileType == 'npy':
        return np.load(filepath)
    if fileType == 'dat':
        if shape is not None:
            Nx, Nz = shape
            Nz = Nz - 2*pml_width
            Nx = Nx - 2*pml_width
        else:
            raise ValueError('when the filetype of vel is .dat, the shape must be specified.')
        with open(filepath, "rb") as f:
            d = struct.unpack("f"*Nx*Nz, f.read(4*Nx*Nz))
            d = np.array(d)
            d = np.reshape(d, (Nx, Nz))
        return d
    # if fileType == 'segy':
    #     with segyio.open(filepath, ignore_geometry=True) as f:
    #         f.mmap()
    #         vel = []
    #         for trace in f.trace:
    #             vel.append(trace.copy())
    #     vel=np.array(vel).T
    #     return vel
    
def diff_using_roll(input, dim=-1, append=True, padding_value=0):

    dim = input.dim() + dim if dim < 0 else dim
    shifts = -1 if append else 1
    rolled_input = torch.roll(input, shifts=shifts, dims=dim)

    # Fill the idex with value padding_value
    index = [slice(None)] * input.dim()
    index[dim] = -1 if append else 0
    rolled_input[tuple(index)] = padding_value

    diff_result = rolled_input - input if append else input-rolled_input
    return diff_result

    
        
def update_cfg(cfg, geom = 'geom', device='cpu'):
    """update the cfg dict, mainly update the Nx and Ny paramters.
    """
    Nx, Ny = cfg[geom]['Nx'], cfg[geom]['Ny']

    if (Nx is None) and (Ny is None) and (cfg[geom]['cPath']):
        vel_path = cfg[geom]['cPath']
        vel = load_file_by_type(vel_path)
        Ny, Nx = vel.shape
    cfg[geom]['_oriNx'] = Nx
    cfg[geom]['_oriNz'] = Ny
    cfg[geom].update({'Nx':Nx + 2*cfg[geom]['pml']['N']})
    cfg[geom].update({'Ny':Ny + 2*cfg[geom]['pml']['N']})
    cfg.update({'domain_shape': (cfg['geom']['Ny'], cfg['geom']['Nx'])})
    cfg.update({'device': device})
    return cfg

    
def get_localrank(host_file, rank=0):
    with open(host_file, "r") as f:
        texts = f.readlines()
    hosts = {}
    for text in texts:
        node, cpu_num = text.split(":")
        hosts[node] = int(cpu_num)
    # Get the ip address of current node
    current_ip_address = socket.gethostbyname(socket.gethostname())
    local_rank = rank%hosts[current_ip_address]
    return hosts, current_ip_address, local_rank

def check_dir(path):
    if not os.path.exists(path):
        print(f"{path} does not exists, trying to make dir...")
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print(e)
            return False
        
def roll(wavelet, data, split=0.2):
    nt = data.shape[0]
    # Calculate time-shifts
    time_shifts = torch.arange(0, split*nt)
    # Randomly assign polarity and time-shift
    p = np.random.randint(1, 3)  # Random positive integer (1 or 2)
    tau_s = np.random.choice(time_shifts)

    # Roll the data along the time axis
    rolled_signal = (-1)**p * np.roll(wavelet, int(tau_s), axis=1)
    rolled_signal[:,0:int(tau_s)] = 0
    rolled_data = (-1)**p * np.roll(data, int(tau_s), axis=0)
    rolled_data[0:int(tau_s)] = 0

    return rolled_signal, rolled_data


