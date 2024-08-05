import torch
import numpy as np
from torch.nn.functional import conv2d
from .utils import restore_boundaries, diff_using_roll
from .convkernel import kernelx, kernely, padding, kernelx_nc, kernely_nc, kernel
from .acoustic_habc import habc
from .convkernel import kernelx_gc_pos, kernelx_gc_neg, kernely_gc_pos, kernely_gc_neg
from .convkernel import kernelx_gc_pos2, kernelx_gc_neg2, kernely_gc_pos2, kernely_gc_neg2

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

def gradient(y, h, kernel):
    operator = (2*h) ** (-1) * kernel.to(y.device)
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=padding).squeeze(1)

def conv(y, kernel):
    y = y.unsqueeze(1)
    return conv2d(y, kernel, padding=padding).squeeze(1)

def _time_step(*args, **kwargs):

    vp, rho = args[0:2]
    p1, p2 = args[2:4]
    dt, h, b = args[4:7]
    habc_masks = kwargs['habcs']

    p_x_pos = conv(p1, kernelx_gc_pos)
    p_x_neg = conv(p1, kernelx_gc_neg)
    p_z_pos = conv(p1, kernely_gc_pos)
    p_z_neg = conv(p1, kernely_gc_neg)

    buoyancy = rho.pow(-1).unsqueeze(0)

    # Hugh D. Geiger and Pat F. Daley(2003) eq.9
    b_x_pos = conv(buoyancy, kernelx_gc_pos2)/2.
    b_x_neg = conv(buoyancy, kernelx_gc_neg2)/2.
    b_z_pos = conv(buoyancy, kernely_gc_pos2)/2.
    b_z_neg = conv(buoyancy, kernely_gc_neg2)/2.

    # Wave equation
    # 10.1190/geo2011-0345.1 EQ. 27
    pnext = 2*p1-p2 + dt**2*h**(-2)*rho*vp**2*(b_x_pos*p_x_pos - b_x_neg*p_x_neg + b_z_pos*p_z_pos - b_z_neg*p_z_neg)

    # apply HABC
    pnext = habc(pnext, p1, p2, vp, b, dt, h, maskidx = habc_masks)

    return pnext, p1

def _time_step_backward(*args, **kwargs):

    vp, rho = args[0:2]
    p1, p2 = args[2:4]
    dt, h, b = args[4:7]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    p_x_pos = conv(p1, kernelx_gc_pos)
    p_x_neg = conv(p1, kernelx_gc_neg)
    p_z_pos = conv(p1, kernely_gc_pos)
    p_z_neg = conv(p1, kernely_gc_neg)

    buoyancy = rho.pow(-1).unsqueeze(0)

    # Hugh D. Geiger and Pat F. Daley(2003) eq.9
    b_x_pos = conv(buoyancy, kernelx_gc_pos2)/2.
    b_x_neg = conv(buoyancy, kernelx_gc_neg2)/2.
    b_z_pos = conv(buoyancy, kernely_gc_pos2)/2.
    b_z_neg = conv(buoyancy, kernely_gc_neg2)/2.

    # Wave equation
    # 10.1190/geo2011-0345.1 EQ. 27
    pnext = 2*p1-p2 + dt**2*h**(-2)*rho*vp**2*(b_x_pos*p_x_pos - b_x_neg*p_x_neg + b_z_pos*p_z_pos - b_z_neg*p_z_neg)

    with torch.no_grad():
        pnext = restore_boundaries(pnext, h_bd)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1

def _time_step_backward_multiple(*args, **kwargs):
    vp, rho = args[0:2]
    p1, p2 = args[2:4]
    dt, h, b = args[4:7]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    p_x_pos = conv(p1, kernelx_gc_pos)
    p_x_neg = conv(p1, kernelx_gc_neg)
    p_z_pos = conv(p1, kernely_gc_pos)
    p_z_neg = conv(p1, kernely_gc_neg)

    buoyancy = rho.pow(-1).unsqueeze(0)

    # Hugh D. Geiger and Pat F. Daley(2003) eq.9
    b_x_pos = conv(buoyancy, kernelx_gc_pos2)/2.
    b_x_neg = conv(buoyancy, kernelx_gc_neg2)/2.
    b_z_pos = conv(buoyancy, kernely_gc_pos2)/2.
    b_z_neg = conv(buoyancy, kernely_gc_neg2)/2.

    # Wave equation
    # 10.1190/geo2011-0345.1 EQ. 27
    pnext = 2*p1-p2 + dt**2*h**(-2)*rho*vp**2*(b_x_pos*p_x_pos - b_x_neg*p_x_neg + b_z_pos*p_z_pos - b_z_neg*p_z_neg)

    with torch.no_grad():
        pnext = restore_boundaries(pnext, h_bd, multiple=True)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1
