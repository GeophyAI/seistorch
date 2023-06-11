import torch
from torch.nn.functional import conv2d
from .utils import restore_boundaries

def _laplacian(y, h):
    """Laplacian operator"""
    kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(y.device)
    operator = h ** (-2) * kernel
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=1).squeeze(1)

def _time_step(*args):

    c = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    # b = 0
    # When b=0, without boundary conditon.
    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2 - b * dt**-1), h2)
                + torch.mul(c.pow(2), _laplacian(h1, h)))
                )
    return y, h1

def _time_step_backward(*args):

    vp = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    b = b.unsqueeze(0)

    # b = 0

    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2 - b * dt**-1), h2)
                + torch.mul(vp.pow(2), _laplacian(h1, h)))
                )

    y = restore_boundaries(y, h_bd)

    #y = src_func(y, src_values, 1)
    # y = y.clone()
    y = src_func(y, src_values, 1)

    return y, h1