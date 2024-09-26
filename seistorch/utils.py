import argparse
import os
import pickle
import jax.numpy as jnp
import jax
from typing import Any, Iterable, List, Tuple
import traceback

import numpy as np
import torch
from prettytable import PrettyTable
from scipy import signal


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

def roll_jax(wavelet, data, split=0.2, key=None):
    nt = data.shape[0]
    # Calculate time-shifts
    time_shifts = jnp.arange(0, split * nt, dtype=jnp.int32)
    # Randomly assign polarity and time-shift
    p = jax.random.randint(key, 1, 1, 3)  # Random positive integer (1 or 2)
    tau_s = jax.random.choice(key, time_shifts)

    # Create a mask to set the first tau_s elements to zero
    mask = jnp.arange(nt) < tau_s

    # Roll the data along the time axis
    rolled_signal = (-1)**p * jnp.roll(wavelet, tau_s, axis=0)
    rolled_signal = rolled_signal * (~mask)
    rolled_data = (-1)**p * jnp.roll(data, tau_s, axis=0)
    rolled_data = rolled_data * (~mask.reshape(-1, 1, 1))

    return rolled_signal, rolled_data

def ricker_wave(fm, dt, T, delay = 80, dtype='tensor', inverse=False):
    """
        Ricker-like wave.
    """
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

    if "numpy" in str(type(x)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).type(dtype)
    elif isinstance(x, float) or isinstance(x, int):
        return torch.tensor(x).type(dtype)
    elif isinstance(x, list):
        if None in x:
            return torch.Tensor([])  # Return empty tensor
        new_list = []
        for item in x:
            if hasattr(item, 'device'):
                new_list.append(item.cpu().numpy())
            else:
                new_list.append(item)
        x = np.array(new_list)
        return torch.from_numpy(x).type(dtype)
    else:
        return torch.from_numpy(x.cpu().numpy()).type(dtype)

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

def nestedlist2tensor(nestedlist):
    """Convert a nested list to a tensor.
    """
    if isinstance(nestedlist, list):
        return torch.stack([nestedlist2tensor(item) for item in nestedlist])
    else:
        return nestedlist