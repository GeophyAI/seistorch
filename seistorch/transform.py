import torch
import numpy as np

def abs(d):
    return torch.abs(d)

def both_nonnegative(d1, d2, type='abs'):
    valid_types = ['abs', 'square', 'linear', 'envelope', 'exp']
    assert type in valid_types, f"type must be one of {valid_types}"
    if type=='abs':
        return torch.abs(d1), torch.abs(d2)
    elif type=='square':
        return d1**2, d2**2
    elif type=='linear':
        dmin = min(torch.min(d1), torch.min(d2))
        value = dmin if dmin<0 else 0
        return d1-value, d2-value
    elif type=='exp':
        beta = 1.1
        return torch.exp(beta*d1), torch.exp(beta*d2)
    elif type=='envelope':
        return envelope(d1), envelope(d2)

def envelope(d):
    return torch.abs(hilbert(d))

def hilbert(data):
    """
    Compute the Hilbert transform of the input data tensor.

    Args:
        data (torch.Tensor): The input data tensor.

    Returns:
        torch.Tensor: The Hilbert transform of the input data tensor.
    """

    nt, _, _ = data.shape #(nsamples, ntraces, nchannels)
    # nfft = 2 ** (nt - 1).bit_length()
    nfft = nt # the scipy implementation uses this

    # Compute the FFT
    data_fft = torch.fft.fft(data, n=nfft, dim=0)

    # Create the filter
    # h = torch.zeros(nfft, device=data.device).unsqueeze(1).unsqueeze(2)
    h = np.zeros(nfft, dtype=np.float32)

    if nfft % 2 == 0:
        h[0] = h[nfft // 2] = 1
        h[1:nfft // 2] = 2
    else:
        h[0] = 1
        h[1:(nfft + 1) // 2] = 2

    h = np.expand_dims(h, 1)
    h = np.expand_dims(h, 2)
    h = torch.from_numpy(h).to(data.device)
    # h = h.requires_grad_(True)
    # Apply the filter and compute the inverse FFT
    hilbert_data = torch.fft.ifft(data_fft * h, dim=0)

    # Truncate the result to the original length
    #hilbert_data = hilbert_data#[:nt]

    return hilbert_data

def integrate(d, dim=0):
    return torch.cumsum(d, dim=dim)

def nonnegative(d, type='abs'):
    valid_types = ['abs', 'square', 'linear']
    assert type in valid_types, f"type must be one of {valid_types}"
    if type=='abs':
        return torch.abs(d)
    elif type=='square':
        return d**2
    elif type=='linear':
        dmin = torch.min(d)
        value = dmin if dmin<0 else 0
        return d-value

def square(d):
    return d**2

def norm(d, dim=0, ntype='sumis1', eps=1e-10):
    """Normalize the trace.

    Args:
        d (Tensor): A torch.Tensor.
        dim (int, optional): The dimension to normalize. Defaults to 0.
        ntype (str, optional): The normalization type. Defaults to 'sumis1'.
        eps (_type_, optional): A small number to avoid division by zero. Defaults to 1e-10.

    Returns:
        Tensor: The normalized trace.
    """
    valid_types = ['sumis1', 'maxis1']
    assert ntype in valid_types, f"ntype must be one of {valid_types}"
    if ntype=='sumis1':
        return d/(d.sum(dim=dim,keepdim=True)+eps)
    elif ntype=='maxis1':
        return d/(d.max(dim=dim,keepdim=True)+eps)