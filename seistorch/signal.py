import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid
from torchaudio.functional import filtfilt
from seistorch.transform import hilbert
from functools import partial
import jax
import jax.numpy as jnp

from torchvision.transforms.functional import gaussian_blur

class SeisSignal:

    def __init__(self, cfg=None, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.__setup__()

    def __setup__(self, ):
        self.dt = self.cfg['geom']['dt']
        self.forder = self.cfg['training']['filter_ord']

    def decide_filter_type(self, freq):
        """Summary: Decide the filter type

        Args:
            freq (in, float, list): The frequency of the filter.

        Returns:
            str : The filter type.
        """
        return decide_filter_type(freq)
    
    def _filter_(self, d, b, a, axis):
        return signal.filtfilt(b, a, d, axis=axis).astype(np.float32)

    def _filter2_(self, sos, data, axis=0, zero_phase=True):
        if zero_phase:
            firstpass = signal.sosfilt(sos, data, axis=axis)
            return signal.sosfilt(sos, firstpass[::-1], axis=axis)[::-1]
        else:
            return signal.sosfilt(sos, data, axis=axis)

    def filter(self, d, freqs, axis=0, threads=1, backend='scipy', **kwargs):
        
        filter_mode = self.decide_filter_type(freqs)

        if self.logger is not None:
            self.logger.print(f"Data filtering (mode: {filter_mode}): frequency:{freqs}")

        if freqs == "all":
            return d
        else:
            valid_modes = ["lowpass", "highpass", "bandpass"]
            assert filter_mode in valid_modes, "mode must be lowpass, highpass or bandpass"

            if filter_mode in ["lowpass", "highpass"]:
                if isinstance(freqs, list): freqs = freqs[0]
                assert isinstance(freqs, (int, float)), "freqs must be a number for lowpass or highpass filter"
                freqs = [freqs]

            if filter_mode =="bandpass":
                assert isinstance(freqs, (list, tuple)), "freqs must be a list or tuple for bandpass filter"

        wn = [2*freq/(1/self.dt) for freq in list(freqs)]
        wn = wn[0] if len(wn)==1 else wn

        nshots  = d.shape[0]

        # call _filter_
        b, a = signal.butter(self.forder, Wn=wn, btype=filter_mode)

        if backend == 'scipy':

            d_filter = np.empty(nshots, dtype=np.ndarray)

            d_filter[:] = Parallel(n_jobs=threads)(
                delayed(self._filter_)(d[i], b, a, axis)
                for i in range(nshots)
            )

            return d_filter
        
        if backend == 'torch':
            # In this case, d is a Tensorlist
            b = torch.from_numpy(b).double().to(d.device)
            a = torch.from_numpy(a).double().to(d.device)

            for i in range(d.shape[0]):
                data  = d.data[i] # (nsample, ntraces, nchannles)
                data = data.permute(1, 2, 0) # (ntraces, nchannels, nsamples)
                data = filtfilt(data.double(), a, b, clamp=False)
                data = data.float().permute(2, 0, 1)
                d.data[i] = data

            return d

    def ricker(self, dtype='tensor', inverse=False):
        """
            Ricker-like wave.
        """

        fm = self.cfg['geom']['fm']
        dt = self.cfg['geom']['dt']
        nt = self.cfg['geom']['nt']
        delay = self.cfg['geom']['wavelet_delay']

        ricker = []
        delay = delay * dt 
        t = np.arange(0, nt*dt, dt)

        c = np.pi * fm * (t - delay) #  delay
        p = -1 if inverse else 1
        ricker = p*(1-2*np.power(c, 2)) * np.exp(-np.power(c, 2))

        if dtype == 'numpy':
            return np.array(ricker).astype(np.float32)
        else:
            return torch.from_numpy(np.array(ricker).astype(np.float32))


def batch_sta_lta(traces, sta_len, lta_len, threshold_on=0.5, threshold_off=1.0):
    """Summary: Calculate the STA/LTA ratio of a signal
    Args:
        traces (np.ndarray): 2D array of traces with shape (nsamples, ntraces)
        sta_len (int): Length of the short-term average window
        lta_len (int): Length of the long-term average window
        threshold_on (float): Threshold for turning on the trigger
        threshold_off (float): Threshold for turning off the trigger
    """
    nt, _ = traces.shape
    sta_kernel = np.ones(sta_len)
    lta_kernel = np.ones(lta_len)
    # Apply convolve along the first axis of the array
    sta = np.apply_along_axis(lambda x: np.convolve(x, sta_kernel, mode='same'), axis=0, arr=traces)
    lta = np.apply_along_axis(lambda x: np.convolve(x, lta_kernel, mode='same'), axis=0, arr=traces)

    ratio = sta / (lta+0.001)

    # Track trigger
    trigger = (ratio > threshold_on).astype(int)
    trigger[1:] -= (ratio[:-1] < threshold_off).astype(int)
    
    # Get first arrival time 
    fa_time = np.argmax(trigger, axis=0)
    fa_time[fa_time==0] = nt-1

    return fa_time

def batch_sta_lta_torch(traces, sta_len, lta_len, threshold_on=0.5, threshold_off=1.0):
    """Summary: Calculate the STA/LTA ratio of a signal

    Args:
        traces (np.ndarray): 2D array of traces with shape (nsamples, ntraces)
        sta_len (int): Length of the short-term average window
        lta_len (int): Length of the long-term average window
        threshold_on (float): Threshold for turning on the trigger
        threshold_off (float): Threshold for turning off the trigger
    """
    def apply_along_axis(function, arr=None, axis: int = 0):
        return torch.stack([
            function(x_i) for x_i in torch.unbind(arr, dim=axis)
        ], dim=axis)
    sta_kernel = torch.ones(1, 1, sta_len, device=traces.device)
    lta_kernel = torch.ones(1, 1, lta_len, device=traces.device)

    # Apply convolve along the first axis of the array
    # conv 1d: input shape: (batch_size, in_channels, signal_len)
    #           kernel shape: (out_channels, in_channels, kernel_size)
    padding = "same"
    # Method 1
    # sta = apply_along_axis(lambda x: torch.nn.functional.conv1d(x.unsqueeze(0), sta_kernel, padding=padding), arr=traces, axis=0)
    # lta = apply_along_axis(lambda x: torch.nn.functional.conv1d(x.unsqueeze(0), lta_kernel, padding=padding), arr=traces, axis=0)
    

    # Method 2
    padding = "same"
    sta = []
    lta = []
    nt, nr = traces.shape
    for i in range(nr):
        sta.append(torch.nn.functional.conv1d(traces[:,i].unsqueeze(0).unsqueeze(0), sta_kernel, padding=padding))
        lta.append(torch.nn.functional.conv1d(traces[:,i].unsqueeze(0).unsqueeze(0), lta_kernel, padding=padding))
    sta = torch.cat(sta, dim=0).permute(2,0,1)
    lta = torch.cat(lta, dim=0).permute(2,0,1)

    ratio = sta / (lta+0.001)
    # Track trigger
    trigger = (ratio > threshold_on).int()
    trigger[1:] -= (ratio[:-1] < threshold_off).int()
    
    # Get first arrival time 
    fa_time = torch.argmax(trigger, axis=0)
    fa_time[fa_time==0] = nt-1

    return fa_time.squeeze().int()

def differentiable_trvaletime_difference(input, beta=1):
    *_, n = input.shape
    input = F.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result

def filter(data, dt=0.001, freqs=[2,5], forder=3, btype='highpass', axis=1):
      
    wn = [2*freq/(1/dt) for freq in list(freqs)]
    wn = wn[0] if len(wn)==1 else wn

    # call _filter_
    b, a = signal.butter(forder, Wn=wn, btype=btype)

    filtered_data = signal.filtfilt(b, a, data, axis=axis).astype(np.float32)
    return filtered_data

def filter_jax(data, dt=0.001, freqs=[2,5], forder=3, axis=1):

    btype = decide_filter_type(freqs)

    if btype == "all":
        return data

    if btype in ["lowpass", "highpass"]:
        if isinstance(freqs, list): freqs = freqs[0]
        assert isinstance(freqs, (int, float)), "freqs must be a number for lowpass or highpass filter"
        freqs = [freqs]

    wn = [2*freq/(1/dt) for freq in list(freqs)]
    wn = wn[0] if len(wn)==1 else wn

    # calculate the filter coefficients
    b, a = signal.butter(forder, Wn=wn, btype=btype)

    # apply the filter
    filtered = jax_filtfilt(b, a, data, axis=axis)

    return filtered

def apply_filter(b, a, x):
    def filter_step(carry, x_t):
        b_, a_, y_t = carry # Get the previous state (historical input and output)
        
        # y[n] = b[0]x[n] + b[1]x[n-1] + ... - a[1]y[n-1] - ...
        new_y_t = jnp.dot(b, jnp.concatenate([jnp.array([x_t]), b_[:-1]])) - jnp.dot(a[1:], a_)
        
        # Update the state
        b_ = jnp.concatenate([jnp.array([x_t]), b_[:-1]])
        a_ = jnp.concatenate([jnp.array([new_y_t]), a_[:-1]])
        
        return (b_, a_, new_y_t), new_y_t

    carry_init = (jnp.zeros(len(b)), jnp.zeros(len(a) - 1), 0.0)
    
    _, y = jax.lax.scan(filter_step, carry_init, x)
    return y

@partial(jax.jit, static_argnums=(3))
def jax_filtfilt(b, a, x, axis=-1):
    # Move the array so that the target axis is the last dimension
    x_moved = jnp.moveaxis(x, axis, -1)
    
    # Apply bidirectional filtering to each subarray along the last dimension
    def filt_along_axis(arr):
        # Forward filtering
        y_fwd = apply_filter(b, a, arr)
        # Backward filtering
        y_bwd = apply_filter(b, a, y_fwd[::-1])
        return y_bwd[::-1]
    
    # Apply the filter to all subarrays, keeping the other dimensions unchanged
    def nested_vmap(f, x):
        ndim = x.ndim
        for _ in range(ndim - 1):
            f = jax.vmap(f)
        return f(x)
    
    y_filtered = nested_vmap(filt_along_axis, x_moved)
    # Restore the axis of the array to its original position
    return jnp.moveaxis(y_filtered, -1, axis)

def decide_filter_type(freq):
    """Summary: Decide the filter type

    Args:
        freq (in, float, list): The frequency of the filter.

    Returns:
        str : The filter type.
    """
    if isinstance(freq, (int, float)): filter_mode = "lowpass"
    if isinstance(freq, list): 
        if len(freq) == 1: filter_mode = "lowpass"
        if len(freq) == 2: filter_mode = "bandpass"
    if freq == "all": filter_mode = "all"
    return filter_mode

def generate_arrival_mask(d, top_win=200, down_win=200):
    mask = torch.zeros_like(d)
    nb, nt, nr, nc = mask.shape
    for idx in range(nb):
        arrival = batch_sta_lta_torch(d[idx, :, :, 0], 100, 500, 0.5, 1)
        # arrival = batch_sta_lta_torch(d[idx, :, :, 0], 200, 1000, 0.5, 1)

        for tno in range(nr):
            _arr = int(arrival[tno])
            top = 0 if _arr-top_win<0 else _arr-top_win
            down = nt if _arr+down_win>nt else _arr+down_win
            mask[idx, :down, tno] = 1
    return mask

def generate_arrival_mask_np(d, top_win=200, down_win=200, swin=100, lwin=500, on=0.5, off=1):
    mask = np.zeros_like(d)
    nb, nt, nr, nc = mask.shape
    for idx in range(nb):
        arrival = batch_sta_lta(d[idx, :, :, 0], swin, lwin, on, off)
        for tno in range(nr):
            _arr = int(arrival[tno])
            top = 0 if _arr-top_win<0 else _arr-top_win
            down = nt if _arr+down_win>nt else _arr+down_win
            mask[idx, :down, tno] = 1
    return mask

def gaussian_filter(input_tensor, sigma, radius, axis=-1):
    """
    Function that applies Gaussian filter to a tensor along a given axis.

    Args:
    - input_tensor: Input tensor
    - sigma: Standard deviation of the Gaussian filter
    - width: Width of the Gaussian filter
    - axis: Axis along which the Gaussian filter is applied

    Returns:
    - output_tensor: Output tensor
    """

    ndim = input_tensor.ndim
    # chech odd or even
    if radius % 2 == 0:
        kernel_size = 2*radius+1
    else:
        kernel_size = 2*radius

    dev = input_tensor.device
    # Generate the Gaussian kernel
    kernel = torch.tensor(
        np.exp(-(np.arange(kernel_size) - kernel_size // 2)**2 / (2 * sigma**2)),
        dtype=torch.float32,
        device=dev
    )
    kernel = kernel / kernel.sum()

    # Reshape the kernel
    kernel = kernel.view(1, 1, -1)

    # Pad the input tensor
    padding = kernel_size // 2
    bound = padding

    # Padding 
    input_numpy = input_tensor.cpu().numpy()

    input_numpy = np.pad(input_numpy, ((padding,padding), (padding, padding)), mode='reflect')

    input_tensor = torch.from_numpy(input_numpy).float().to(dev)

    # Apply the Gaussian filter
    if ndim == 1:
        result = F.conv1d(input_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=padding, stride=1)
    elif ndim == 2:
        if axis == 0:
            kernel = kernel.view(1, 1, -1, 1)
            padding = (padding, 0)
        elif axis == 1:
            kernel = kernel.view(1, 1, 1, -1)
            padding = (0, padding)
        result = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=padding, stride=1)
        if bound > 0:
            result = result.squeeze(0).squeeze(0)[bound:-bound, bound:-bound]
        else:
            result = result.squeeze(0).squeeze(0)
    elif ndim == 3:
        if axis == 0:
            kernel = kernel.view(1, 1, 1, -1, 1)
            padding = (0, padding, 0)
        elif axis == 1:
            kernel = kernel.view(1, 1, 1, 1, -1)
            padding = (0, 0, padding)
        elif axis == 2:
            kernel = kernel.view(1, 1, -1, 1, 1)
            padding = (padding, 0, 0)
        result = F.conv3d(input_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=padding, stride=1)
    else:
        raise ValueError("Unsupported number of dimensions. Only 1D, 2D, and 3D are supported.")
    return result

def generate_mask(fa_time, nt, nr, N):
    mask = torch.zeros(nt, nr, dtype=torch.float32)

    start = torch.maximum(torch.zeros_like(fa_time), fa_time.int()-N//2)
    end = torch.minimum(torch.ones_like(fa_time)*nt, fa_time.int()+N//2)
    row_indices = torch.arange(nt).unsqueeze(1)

    mask_indices = (row_indices >= start.unsqueeze(0)) & (row_indices <= end.unsqueeze(0))

    mask[mask_indices] = 1

    return mask

def integrate(d):
    return cumulative_trapezoid(d, dx=1, initial=0)

def instantaneous_phase(data):
    """
    Compute the instantaneous phase of the input data tensor.

    Args:
        data (torch.Tensor): The input data tensor with shape (time_samples, num_traces, num_channels).

    Returns:
        torch.Tensor: The instantaneous phase of the input data tensor.
    """
    # Compute the instantaneous phase
    hilbert_d = hilbert(data)
    imag, real = hilbert_d.imag, hilbert_d.real
    ip = torch.arctan2(imag, real)
    return ip

def local_coherence(x, y, wt=101, wx=11, sigma_tau=21.0, sigma_hx=11.0):
    """Local coherence between two batched seismic data

    Args:
        x (torch.Tensor): A torch.Tensor. shape: (batch_size, time_samples, num_traces, num_channels)
        y (torch.Tensor): Other torch.Tensor with the same shape as x.
        wt (int, optional): Width of window along time axis. Defaults to 101.
        wx (int, optional): Width of window along trace axis. Defaults to 11.
        sigma_tau (float, optional): Sigma of gaussian kernel. Defaults to 21.0.
        sigma_hx (float, optional): Sigma of gaussian kernel. Defaults to 11.0.

    Returns:
        Tensor: The local coherence between x and y with shape (batch_size, time_samples, num_traces, channels)
    """
    half_window_tau = wt // 2
    half_window_hx = wx // 2
    # Permute the tensors to (batch_size, num_channels, time_samples, num_traces)
    syn = x.permute(0, 3, 1, 2)
    obs = y.permute(0, 3, 1, 2)
    
    # Convolve with Gaussian kernel
    window_syn = gaussian_blur(syn, (wt, wx), (sigma_tau, sigma_hx))
    window_obs = gaussian_blur(obs, (wt, wx), (sigma_tau, sigma_hx))

    # Remove the mean
    # akernel = average_kernel.to(dev)
    # window_syn = window_syn - F.conv2d(window_syn, akernel, padding=(self.half_window_tau, self.half_window_hx))
    # window_obs = window_obs - F.conv2d(window_obs, akernel, padding=(self.half_window_tau, self.half_window_hx))
    
    # Compute the local coherence
    # Unfold the tensors to get sliding windows
    window_syn_unf = F.unfold(window_syn, (wt, wx), padding=(half_window_tau, half_window_hx))
    window_obs_unf = F.unfold(window_obs, (wt, wx), padding=(half_window_tau, half_window_hx))

    # Compute cosine similarity
    cs = F.cosine_similarity(window_syn_unf, window_obs_unf, dim=1, eps=1e-8)
    cs = cs.view(*syn.shape)
    return cs

def normalize_trace_max(d):
    """Normalize the trace by its maximum value

    Args:
        d (Tensor): A torch.Tensor.
    """
    w = torch.max(torch.abs(d), dim=0, keepdim=True)[0]
    return d / w

def pick_first_arrivals(d, *args, **kwargs):
    _, ntraces, nchannels = d.shape
    fa_times = []
    for c in range(nchannels):
        fa_times.append(batch_sta_lta_torch(d[:, :, c], *args, **kwargs).view(ntraces, 1))
    return torch.cat(fa_times, dim=1)

def pick_first_arrivals_numpy(d, *args, **kwargs):
    _, ntraces, nchannels = d.shape
    fa_times = []
    for c in range(nchannels):
        fa_times.append(batch_sta_lta(d[:, :, c], *args, **kwargs).reshape(-1, 1))
    return np.array(fa_times).T

def ricker_wave(fm, dt, nt, delay = 80, dtype='tensor', inverse=False):
    """
        Ricker-like wave.
    """
    print(f"Wavelet inverse:{inverse}")
    ricker = []
    delay = delay * dt 
    t = np.arange(0, nt*dt, dt)

    c = np.pi * fm * (t - delay) #  delay
    p = -1 if inverse else 1
    ricker = p*(1-2*np.power(c, 2)) * np.exp(-np.power(c, 2))

    if dtype == 'numpy':
        return np.array(ricker).astype(np.float32)
    else:
        return torch.from_numpy(np.array(ricker).astype(np.float32))

def travel_time_diff(x, y, dt=0.001, eps=0):
    if torch.max(torch.abs(x))>eps or torch.max(torch.abs(y))>eps:
        nt = x.shape[0]
        padding = nt-1
        cc = torch.abs(F.conv1d(x.unsqueeze(0), y.unsqueeze(0).unsqueeze(0), padding=padding))
        return (torch.argmax(cc)-nt+1)*dt
    else:
        return 0
