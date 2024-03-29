import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid
from torchaudio.functional import filtfilt

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
        if isinstance(freq, (int, float)): filter_mode = "lowpass"
        if isinstance(freq, list): filter_mode = "bandpass"
        if freq == "all": filter_mode = "all"
        return filter_mode
    
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
                data = filtfilt(data.double(), a, b)
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

def generate_mask(fa_time, nt, nr, N):
    mask = torch.zeros(nt, nr, dtype=torch.float32)

    start = torch.maximum(torch.zeros_like(fa_time), fa_time.int()-N//2)
    end = torch.minimum(torch.ones_like(fa_time)*nt, fa_time.int()+N//2)
    row_indices = torch.arange(nt).unsqueeze(1)

    mask_indices = (row_indices >= start.unsqueeze(0)) & (row_indices <= end.unsqueeze(0))

    mask[mask_indices] = 1

    return mask

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

def differentiable_trvaletime_difference(input, beta=1):
    *_, n = input.shape
    input = F.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result      

def normalize_trace_max(d):
    """Normalize the trace by its maximum value

    Args:
        d (Tensor): A torch.Tensor.
    """
    w = torch.max(torch.abs(d), dim=0, keepdim=True)[0]
    return d / w

def integrate(d):
    return cumulative_trapezoid(d, dx=1, initial=0)