import torch
import numpy as np
from torch.nn.functional import conv2d
from .utils import restore_boundaries
from .convkernel import kernelx, kernely, padding
from .convkernel import kernel as kernel_xy
from .acoustic_habc import habc

def _laplacian(y, h, kernel):
    """Laplacian operator"""
    # kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(y.device)
    # kernel = torch.tensor([[[[0.0, 0.0, -0.083, 0.0, 0.0],
    #                          [0.0, 0.0, 1.333, 0.0, 0.0],
    #                          [-0.083, 1.333, -2.5, 1.333, -0.083],
    #                          [0.0, 0.0, 1.333, 0.0, 0.0],
    #                          [0.0, 0.0, -0.083, 0.0, 0.0]]]]).to(y.device)
    operator = h ** (-2) * kernel.to(y.device)
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=padding).squeeze(1)

def _time_step(*args, **kwargs):

    vp, eps, delta = args[0:3]
    p1, p2, q1, q2 = args[3:7]
    dt, h, b = args[7:10]
    habc_masks = kwargs['habcs']

    nablax_p1 = _laplacian(p1, h, kernelx)
    nablaz_p1 = _laplacian(p1, h, kernely)
    nablax_q1 = _laplacian(q1, h, kernelx)
    nablaz_q1 = _laplacian(q1, h, kernely)

    vp2dt2 = vp**2*dt**2

    # 10.3997/2214-4609.201402310 eq (9)

    pnext = 2*p1-p2 + ((1+2*delta)*(nablax_p1+nablax_q1)+nablaz_p1)*vp2dt2

    # 10.3997/2214-4609.201402310 eq (10)
    qnext = 2*q1-q2 + (2*(eps-delta)*(nablax_p1+nablax_q1))*vp2dt2

    # apply HABC
    pnext = habc(pnext, p1, p2, vp, b, dt, h, maskidx = habc_masks)
    qnext = habc(qnext, q1, q2, vp, b, dt, h, maskidx = habc_masks)

    return pnext, p1, qnext, q1

def _time_step_backward(*args, **kwargs):
    bwidth=50
    N=1
    vp = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    vp = vp.unsqueeze(0)
    b = b.unsqueeze(0)

    y = torch.mul((dt**-2).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2), h2)
                + torch.mul(vp.pow(2), _laplacian(h1, h)))
                )

    with torch.no_grad():
        y = restore_boundaries(y, h_bd)

    y = src_func(y, src_values, 1)

    return y, h1

def _time_step_backward_multiple(*args, **kwargs):

    vp = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    vp = vp.unsqueeze(0)
    b = b.unsqueeze(0)

    y = torch.mul((dt**-2).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2), h2)
                + torch.mul(vp.pow(2), _laplacian(h1, h)))
                )
    
    with torch.no_grad():
        y = restore_boundaries(y, h_bd, multiple=True)
    
    y = src_func(y, src_values, 1)

    return y, h1