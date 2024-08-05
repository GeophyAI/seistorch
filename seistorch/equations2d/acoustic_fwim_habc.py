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
    vp, rx, rz = args[0:3]
    p1, p2 = args[3:5]
    dt, h, b = args[5:8]
    habc_masks = kwargs['habcs']

    # Centered finite difference
    v_x = gradient(vp.unsqueeze(0), h, kernelx_nc)
    v_z = gradient(vp.unsqueeze(0), h, kernely_nc)
    p_x = gradient(p1, h, kernelx_nc)
    p_z = gradient(p1, h, kernely_nc)

    term1 = vp**2*dt**2*_laplacian(p1, h, kernel)
    term2 = vp*dt**2*(v_x*p_x+v_z*p_z)
    term3 = 2*vp**2*dt**2*(rx*p_x+rz*p_z)

    # Forward & Backward Difference
    # v_x_pos = conv(vp.unsqueeze(0), kernelx_gc_pos2)/2.
    # v_x_neg = conv(vp.unsqueeze(0), kernelx_gc_neg2)/2.
    # v_z_pos = conv(vp.unsqueeze(0), kernely_gc_pos2)/2.
    # v_z_neg = conv(vp.unsqueeze(0), kernely_gc_neg2)/2.

    # p_x_pos = conv(p1, kernelx_gc_pos)
    # p_x_neg = conv(p1, kernelx_gc_neg)
    # p_z_pos = conv(p1, kernely_gc_pos)
    # p_z_neg = conv(p1, kernely_gc_neg)

    # term1 = vp**2*dt**2*_laplacian(p1, h, kernel)
    # term2 = vp*dt**2*h**(-2)*(v_x_pos*p_x_pos-v_x_neg*p_x_neg+v_z_pos*p_z_pos-v_z_neg*p_z_neg)
    # term3 = 2*vp**2*dt**2*h**(-1)*(rx*p_x_pos+rz*p_z_pos)
    # Wave equation
    pnext = 2*p1-p2 + term1 + term2 - term3

    # apply HABC
    pnext = habc(pnext, p1, p2, vp, b, dt, h, maskidx = habc_masks)

    return pnext, p1

def _time_step_backward(*args, **kwargs):

    vp, rx, rz = args[0:3]
    p1, p2 = args[3:5]
    dt, h, b = args[5:8]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)

    # Centered finite difference
    v_x = gradient(vp, h, kernelx_nc)
    v_z = gradient(vp, h, kernely_nc)
    p_x = gradient(p1, h, kernelx_nc)
    p_z = gradient(p1, h, kernely_nc)

    term1 = vp**2*dt**2*_laplacian(p1, h, kernel)
    term2 = vp*dt**2*(v_x*p_x+v_z*p_z)
    term3 = 2*vp**2*dt**2*(rx*p_x+rz*p_z)
    # v_x_pos = conv(vp, kernelx_gc_pos2)/2.
    # v_x_neg = conv(vp, kernelx_gc_neg2)/2.
    # v_z_pos = conv(vp, kernely_gc_pos2)/2.
    # v_z_neg = conv(vp, kernely_gc_neg2)/2.

    # p_x_pos = conv(p1, kernelx_gc_pos)
    # p_x_neg = conv(p1, kernelx_gc_neg)
    # p_z_pos = conv(p1, kernely_gc_pos)
    # p_z_neg = conv(p1, kernely_gc_neg)

    # term1 = vp**2*dt**2*_laplacian(p1, h, kernel)
    # term2 = vp*dt**2*h**(-2)*(v_x_pos*p_x_pos-v_x_neg*p_x_neg+v_z_pos*p_z_pos-v_z_neg*p_z_neg)
    # term3 = 2*vp**2*dt**2*h**(-1)*(rx*p_x_pos+rz*p_z_pos)
    # Wave equation
    pnext = 2*p1-p2 + term1 + term2 - term3

    with torch.no_grad():
        pnext = restore_boundaries(pnext, h_bd)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1

def _time_step_backward_multiple(*args, **kwargs):

    vp, rx, rz = args[0:3]
    p1, p2 = args[3:5]
    dt, h, b = args[5:8]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)

    # Centered finite difference
    v_x = gradient(vp, h, kernelx_nc)
    v_z = gradient(vp, h, kernely_nc)
    p_x = gradient(p1, h, kernelx_nc)
    p_z = gradient(p1, h, kernely_nc)

    term1 = vp**2*dt**2*_laplacian(p1, h, kernel)
    term2 = vp*dt**2*(v_x*p_x+v_z*p_z)
    term3 = 2*vp**2*dt**2*(rx*p_x+rz*p_z)
    
    # vp = vp.unsqueeze(0)

    # v_x_pos = conv(vp, kernelx_gc_pos2)/2.
    # v_x_neg = conv(vp, kernelx_gc_neg2)/2.
    # v_z_pos = conv(vp, kernely_gc_pos2)/2.
    # v_z_neg = conv(vp, kernely_gc_neg2)/2.

    # p_x_pos = conv(p1, kernelx_gc_pos)
    # p_x_neg = conv(p1, kernelx_gc_neg)
    # p_z_pos = conv(p1, kernely_gc_pos)
    # p_z_neg = conv(p1, kernely_gc_neg)

    # term1 = vp**2*dt**2*_laplacian(p1, h, kernel)
    # term2 = vp*dt**2*h**(-2)*(v_x_pos*p_x_pos-v_x_neg*p_x_neg+v_z_pos*p_z_pos-v_z_neg*p_z_neg)
    # term3 = 2*vp**2*dt**2*h**(-1)*(rx*p_x_pos+rz*p_z_pos)
    # Wave equation
    pnext = 2*p1-p2 + term1 + term2 - term3

    with torch.no_grad():
        pnext = restore_boundaries(pnext, h_bd, multiple=True)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1
