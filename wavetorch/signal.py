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