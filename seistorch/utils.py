import argparse
import contextlib
import os
import pickle
import socket
import struct
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch.nn.functional as F
import torch
from prettytable import PrettyTable
from scipy import signal
from joblib import Parallel, delayed

def accuracy_onehot(y_pred, y_label):
    """Compute the accuracy for a onehot
    """
    return (y_pred.argmax(dim=1) == y_label).float().mean().item()

def check_dir(path):
    if not os.path.exists(path):
        print(f"{path} does not exists, trying to make dir...")
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print(e)
            return False
        

def cpu_fft(d, dt, N = 5, low = 5, if_plot = True, axis = -1, mode = 'lowpass'):
    """
        implementation of fft.
    """
    if low == "all":
        return d
    else:
        wn = 2*low/(1/dt)
        b, a = signal.butter(N, wn, mode)
        d_filter = signal.filtfilt(b, a, d, axis = axis)
        return d_filter.astype(np.float32)
    
def dict2table(dict_data: dict, table: PrettyTable = None):
    """Convert a dict to a table"""
    _tmp_table = PrettyTable(["Configures", "Value"]) if table is None else table
    tables = []
    for k, v in dict_data.items():
        if isinstance(v, dict):
            #table = PrettyTable(["Configures", "Value"])
            _tables = dict2table(v, PrettyTable([k, "Value"]))
            # for kk, vv in v.items():
            #     table.add_row([kk, vv])
            tables.extend(_tables)
        else:
            _tmp_table.add_row([k, v])
    tables.append(_tmp_table)
    return tables

def diff_using_roll(input, dim=-1, forward=True, padding_value=0):

    def forward_diff(x, dim=-1, padding_value=0):
        """
        Compute the forward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The forward difference of the input tensor.
        """
        diff = x - torch.roll(x, shifts=1, dims=dim)
        diff[..., 0] = padding_value  # pad with specified value
        return diff

    def backward_diff(x, dim=-1, padding_value=0):
        """
        Compute the backward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The backward difference of the input tensor.
        """
        diff = torch.roll(x, shifts=-1, dims=dim) - x
        diff[..., -1] = padding_value  # pad with specified value
        return diff

    if forward:
        return forward_diff(input, dim=dim)
    else:
        return backward_diff(input, dim=dim)

def get_src_and_rec(cfg):
    assert os.path.exists(cfg["geom"]["sources"]), "Cannot found source file."
    assert os.path.exists(cfg["geom"]["receivers"]), "Cannot found receiver file."
    source_locs = read_pkl(cfg["geom"]["sources"])
    recev_locs = read_pkl(cfg["geom"]["receivers"])
    assert len(source_locs)==len(recev_locs), \
        "The lenght of sources and recev_locs must be equal."
    return source_locs, recev_locs

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

def intersection(arrays):
    if not arrays:
        return []

    # 将数组列表转换为NumPy数组
    arrays_np = np.array(arrays)
    # 使用逻辑与运算计算交集
    intersection_result = np.any(arrays_np, axis=0).astype(int)
    
    return intersection_result

def _low_pass_filter(d, b, a, axis):
    return signal.filtfilt(b, a, d, axis=axis).astype(np.float32)

def low_pass(d, dt, N = 5, low = 5, axis = -1, threads=1):
    """
        implementation of low pass filter.
    """
    if low == "all":
        return d
    else:
        wn = 2*low/(1/dt)
        b, a = signal.butter(N, wn, 'lowpass')
        nshots  = d.shape[0]
        d_filter = np.empty(nshots, dtype=np.ndarray)

        d_filter[:] = Parallel(n_jobs=threads)(
            delayed(_low_pass_filter)(d[i], b, a, axis)
            for i in range(nshots)
        )

        # serial version
        #for i in range(nshots):
        #    d_filter[i] = signal.filtfilt(b, a, d[i], axis = axis).astype(np.float32)
        return d_filter
   
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
    
def normalize_power(X):
    return X / torch.sum(X, dim=1, keepdim=True)
    
def pad_by_value(d, pad, mode = 'double'):
    """pad the input by <pad>
    """
    if mode == 'double':
        return d + 2*pad
    else:
        return d + pad
    
def read_pkl(path: str):
    # Open the file in binary mode and load the list using pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

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

def roll2(wavelet, data, mask, split=0.2):
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
    rolled_mask = np.roll(mask, int(tau_s), axis=0)
    rolled_mask[0:int(tau_s)] = 0

    return rolled_signal, rolled_data, rolled_mask

def ricker_wave(fm, dt, T, delay = 80, dtype='tensor', inverse=False):
    """
        Ricker-like wave.
    """
    print(f"Wavelet inverse:{inverse}")
    ricker = []
    delay = delay * dt 
    for i in range(T):
        c = np.pi * fm * (i * dt - delay) #  delay
        p = -1 if inverse else 1
        temp = p*(1-2*np.power(c, 2)) * np.exp(-np.power(c, 2))
        ricker.append(temp)
    if dtype == 'numpy':
        return np.array(ricker).astype(np.float32)
    else:
        return torch.from_numpy(np.array(ricker).astype(np.float32))

def set_dtype(dtype=None):
    if dtype == 'float32' or dtype is None:
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError('Unsupported data type: %s; should be either float32 or float64' % dtype)

def to_tensor(x, dtype=None):
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).type(dtype)
    elif isinstance(x, float):
        return torch.tensor(x).type(dtype)
    else:
        x = np.array(x)
        return torch.from_numpy(x).type(dtype)

def update_cfg(cfg, geom = 'geom', device='cpu'):
    """update the cfg dict, mainly update the Nx and Ny paramters.
    """
    Nx, Ny, Nz = cfg[geom]['Nx'], cfg[geom]['Ny'], cfg[geom]['Nz']

    cfg[geom]['_oriNx'] = Nx
    cfg[geom]['_oriNz'] = Ny
    cfg[geom]['_oriNz'] = Nz

    cfg[geom].update({'Nx':Nx + 2*cfg[geom]['pml']['N']})

    
    if Nz==0: # 2d = (Nz==0)
        if cfg[geom]['multiple']:
            nz = Ny + cfg[geom]['pml']['N']
        else:
            nz = Ny + 2*cfg[geom]['pml']['N']
        cfg[geom].update({'Ny': nz})
        cfg.update({'domain_shape': (cfg[geom]['Ny'], cfg[geom]['Nx'])})
    if Nz>0:

        cfg[geom].update({'Nx':Nx + 2*cfg[geom]['pml']['N']})
        cfg[geom].update({'Ny':Ny + 2*cfg[geom]['pml']['N']})
        cfg[geom].update({'Nz':Nz + 2*cfg[geom]['pml']['N']})
        cfg.update({'domain_shape': (cfg['geom']['Nz'], cfg['geom']['Nx'], cfg['geom']['Ny'])})

    cfg.update({'device': device})
    return cfg

def window_data(X, window_length):
    """Window the sample, X, to a length of window_length centered at the middle of the original sample
    """
    return X[int(len(X) / 2 - window_length / 2):int(len(X) / 2 + window_length / 2)]

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for item in values:
            key, value = item.split('=')
            try:
                value = float(value)
            except ValueError:
                pass
            d[key] = value
        setattr(namespace, self.dest, d)

class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes 
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)

def merge_dicts_with_same_keys(dict_list):
    """Merge the dict with same keys.

    Args:
        dict_list (list): A list of dict.

    Returns:
        dict: The merged dict.
    """
    merged_dict = {}
    for d in dict_list:
        for k, v in d.items():
            if k not in merged_dict:
                merged_dict[k] = []
            merged_dict[k].append(v)
    return merged_dict



interp1d = Interp1d.apply
