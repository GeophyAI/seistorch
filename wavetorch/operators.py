import torch
from torch.nn.functional import conv2d


def _laplacian(y, h):
    """Laplacian operator"""
    kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(y.device)
    operator = h ** (-2) * kernel
    y = y.unsqueeze(1)
    # y = pad(y,pad=(0,0,1,1), mode='circular')
    # y = pad(y,pad=(1,1,0,0),mode='circular')
    return conv2d(y, operator, padding=1).squeeze(1)

def _laplacian2(y, h, k):
    """Laplacian operator"""
    operator = h ** (-2) * k
    y = y.unsqueeze(1)
    # y = pad(y,pad=(0,0,1,1), mode='circular')
    # y = pad(y,pad=(1,1,0,0),mode='circular')
    return conv2d(y, operator, padding=1).squeeze(1)
