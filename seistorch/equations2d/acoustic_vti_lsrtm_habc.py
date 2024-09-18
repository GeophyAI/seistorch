import torch
import numpy as np
from torch.nn.functional import conv2d
from .utils import restore_boundaries
from .convkernel import kernelx, kernely, padding, kernelx_nc, kernely_nc
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

def gradient(y, h, kernel):
    operator = (2*h) ** (-1) * kernel.to(y.device)
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=padding).squeeze(1)

def _time_step(*args, **kwargs):
    
    vp, eps, delta, m = args[0:4]
    p1, p2, sp1, sp2 = args[4:8]
    dt, h, b = args[8:11]
    habc_masks = kwargs['habcs']

    # Method 1: solver in time domain
    # dpdx = gradient(p1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    # dpdz = gradient(p1, h, kernely_nc) # 10.1190/geo2022-0292.1 EQ(21)
    # numerator = -2*(eps-delta)*dpdx**2*dpdz**2
    # denominator = (1+2*eps)*dpdx**4+dpdz**4+2*(1+delta)*dpdx**2*dpdz**2
    # sd = numerator*((denominator+1e-26)**-1)

    # Method2: solver in frequency domain
    # 10.1190/geo2022-0292.1 EQ(12)
    shape = p1.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(p1.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(p1.device)
    k_x, k_y = torch.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*k_x**2*k_y**2
    denominator = (1+2*eps)*k_x**4+k_y**4+2*(1+delta)*k_x**2*k_y**2
    sk = numerator*((denominator+1e-26)**-1)
    sk = sk.real

    vp2dt2 = vp**2*dt**2
    
    # Background wavefield
    nabla_x = _laplacian(p1, h, kernelx)
    nabla_z = _laplacian(p1, h, kernely)
    # 10.1190/geo2022-0292.1 EQ(22)
    ani_laplace_p0 = vp2dt2*((1+2*eps)+sk)*nabla_x + vp2dt2*(1+sk)*nabla_z
    pnext = 2*p1-p2 + ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = _laplacian(sp1, h, kernelx)
    nabla_z_s = _laplacian(sp1, h, kernely)
    spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)+sk)*nabla_x_s + vp2dt2*(1+sk)*nabla_z_s+m*ani_laplace_p0

    # apply HABC
    pnext = habc(pnext, p1, p2, vp, b, dt, h, maskidx = habc_masks)
    spnext = habc(spnext, sp1, sp2, vp, b, dt, h, maskidx = habc_masks)

    return pnext, p1, spnext, sp1

def _time_step_backward(*args, **kwargs):

    vp, eps, delta, m = args[0:4]
    p1, p2, sp1, sp2 = args[4:8]
    dt, h, b = args[8:11]
    h_bd, _, sh_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    # Method 1: solver in time domain
    # dpdx = gradient(p1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    # dpdz = gradient(p1, h, kernely_nc) # 10.1190/geo2022-0292.1 EQ(21)
    # numerator = -2*(eps-delta)*dpdx**2*dpdz**2
    # denominator = (1+2*eps)*dpdx**4+dpdz**4+2*(1+delta)*dpdx**2*dpdz**2
    # sd = numerator*((denominator+1e-26)**-1)
    # when sd=0., the gradient of vp can be calculated
    # when sd!=0, the gradient of al paras cannot be calculated

    # Method2: solver in frequency domain
    # 10.1190/geo2022-0292.1 EQ(12)
    shape = p1.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(p1.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(p1.device)
    k_x, k_y = torch.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*k_x**2*k_y**2
    denominator = (1+2*eps)*k_x**4+k_y**4+2*(1+delta)*k_x**2*k_y**2
    sk = numerator*((denominator+1e-26)**-1)
    sk = sk.real

    vp2dt2 = vp**2*dt**2
    
    # Background wavefield
    nabla_x = _laplacian(p1, h, kernelx)
    nabla_z = _laplacian(p1, h, kernely)
    # 10.1190/geo2022-0292.1 EQ(22)
    ani_laplace_p0 = vp2dt2*((1+2*eps)+sk)*nabla_x + vp2dt2*(1+sk)*nabla_z
    pnext = 2*p1-p2 + ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = _laplacian(sp1, h, kernelx)
    nabla_z_s = _laplacian(sp1, h, kernely)
    spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)+sk)*nabla_x_s + vp2dt2*(1+sk)*nabla_z_s+m*ani_laplace_p0
    # with torch.no_grad():
    pnext = restore_boundaries(pnext, h_bd)
    spnext = restore_boundaries(spnext, sh_bd)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1, spnext, sp1

def _time_step_backward_multiple(*args, **kwargs):

    vp, eps, delta, m = args[0:4]
    p1, p2, sp1, sp2 = args[4:8]
    dt, h, b = args[8:11]
    h_bd, _, sh_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    # Method 1: solver in time domain
    # dpdx = gradient(p1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    # dpdz = gradient(p1, h, kernely_nc) # 10.1190/geo2022-0292.1 EQ(21)
    # numerator = -2*(eps-delta)*dpdx**2*dpdz**2
    # denominator = (1+2*eps)*dpdx**4+dpdz**4+2*(1+delta)*dpdx**2*dpdz**2
    # sd = numerator*((denominator+1e-26)**-1)
    # when sd=0., the gradient of vp can be calculated
    # when sd!=0, the gradient of al paras cannot be calculated

    # Method2: solver in frequency domain
    # 10.1190/geo2022-0292.1 EQ(12)
    shape = p1.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(p1.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(p1.device)
    k_x, k_y = torch.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*k_x**2*k_y**2
    denominator = (1+2*eps)*k_x**4+k_y**4+2*(1+delta)*k_x**2*k_y**2
    sk = numerator*((denominator+1e-26)**-1)
    sk = sk.real

    vp2dt2 = vp**2*dt**2
    
    # Background wavefield
    nabla_x = _laplacian(p1, h, kernelx)
    nabla_z = _laplacian(p1, h, kernely)
    # 10.1190/geo2022-0292.1 EQ(22)
    ani_laplace_p0 = vp2dt2*((1+2*eps)+sk)*nabla_x + vp2dt2*(1+sk)*nabla_z
    pnext = 2*p1-p2 + ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = _laplacian(sp1, h, kernelx)
    nabla_z_s = _laplacian(sp1, h, kernely)
    spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)+sk)*nabla_x_s + vp2dt2*(1+sk)*nabla_z_s+m*ani_laplace_p0
    
    # with torch.no_grad():
    pnext = restore_boundaries(pnext, h_bd, multiple=True)
    spnext = restore_boundaries(spnext, sh_bd, multiple=True)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1, spnext, sp1