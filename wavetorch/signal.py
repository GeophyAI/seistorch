import torch
import numpy as np

def batch_sta_lta(traces, sta_len, lta_len, threshold_on=0.5, threshold_off=1.0):
    """Summary: Calculate the STA/LTA ratio of a signal
    Args:
        traces (np.ndarray): 2D array of traces with shape (nsamples, ntraces)
        sta_len (int): Length of the short-term average window
        lta_len (int): Length of the long-term average window
        threshold_on (float): Threshold for turning on the trigger
        threshold_off (float): Threshold for turning off the trigger
    """
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
    #nsamples, _, = traces.shape
    #traces = traces.view(nsamples, 1, -1)
    def apply_along_axis(function, arr=None, axis: int = 0):
        return torch.stack([
            function(x_i) for x_i in torch.unbind(arr, dim=axis)
        ], dim=axis)
    sta_kernel = torch.ones(1, 1, sta_len, device=traces.device)
    lta_kernel = torch.ones(1, 1, lta_len, device=traces.device)

    # Apply convolve along the first axis of the array
    # conv 1d: input shape: (batch_size, in_channels, signal_len)
    #           kernel shape: (out_channels, in_channels, kernel_size)
    sta = apply_along_axis(lambda x: torch.nn.functional.conv1d(x.unsqueeze(0), sta_kernel, padding='same'), arr=traces, axis=0)
    lta = apply_along_axis(lambda x: torch.nn.functional.conv1d(x.unsqueeze(0), lta_kernel, padding='same'), arr=traces, axis=0)

    ratio = sta / (lta+0.001)

    # Track trigger
    trigger = (ratio > threshold_on).int()
    trigger[1:] -= (ratio[:-1] < threshold_off).int()
    
    # Get first arrival time 
    fa_time = torch.argmax(trigger, axis=0)
    
    return fa_time.squeeze().int()