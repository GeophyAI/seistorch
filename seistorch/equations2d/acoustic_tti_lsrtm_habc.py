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

    vp, eps, delta, theta, m = args[0:5]
    p1, p2, sp1, sp2 = args[5:9]
    dt, h, b = args[9:12]
    habc_masks = kwargs['habcs']

    # from degree to radian
    theta = torch.deg2rad(theta)
    sin0 = torch.sin(theta)
    cos0 = torch.cos(theta)
    sin20 = torch.sin(2*theta)

    nabla_x = _laplacian(p1, h, kernelx)
    nabla_z = _laplacian(p1, h, kernely)

    dpdx = gradient(p1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz = gradient(dpdx, h, kernely_nc)

    # 10.1190/geo2022-0292.1 EQ(A-5)
    shape = p1.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(p1.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(p1.device)
    k_x, k_z = torch.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    denominator = (1+2*eps)*(k_x*cos0-k_z*sin0)**4+(k_x*sin0+k_z*cos0)**4+2*(1+delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    sd = numerator*((denominator+1e-26)**-1)
    sd = sd.real

    vp2dt2 = vp**2*dt**2

    # Background wavefield
    # 10.1190/geo2022-0292.1 EQ(A-7)
    ani_laplace_p0 = vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z-2*eps*vp2dt2*sin20*dpdxdz
    pnext = 2*p1-p2 + ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = _laplacian(sp1, h, kernelx)
    nabla_z_s = _laplacian(sp1, h, kernely)

    dpdx_s = gradient(sp1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz_s = gradient(dpdx_s, h, kernely_nc)
    spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x_s + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z_s-2*eps*vp2dt2*sin20*dpdxdz_s+m*ani_laplace_p0

    # apply HABC
    pnext = habc(pnext, p1, p2, vp, b, dt, h, maskidx = habc_masks)
    spnext = habc(spnext, sp1, sp2, vp, b, dt, h, maskidx = habc_masks)

    return pnext, p1, spnext, sp1

def _time_step_backward(*args, **kwargs):

    vp, eps, delta, theta, m = args[0:5]
    p1, p2, sp1, sp2 = args[5:9]
    dt, h, b = args[9:12]
    h_bd, _, sh_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    # from degree to radian
    theta = torch.deg2rad(theta)
    sin0 = torch.sin(theta)
    cos0 = torch.cos(theta)
    sin20 = torch.sin(2*theta)

    nabla_x = _laplacian(p1, h, kernelx)
    nabla_z = _laplacian(p1, h, kernely)

    dpdx = gradient(p1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz = gradient(dpdx, h, kernely_nc)

    # 10.1190/geo2022-0292.1 EQ(A-5)
    shape = p1.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(p1.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(p1.device)
    k_x, k_z = torch.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    denominator = (1+2*eps)*(k_x*cos0-k_z*sin0)**4+(k_x*sin0+k_z*cos0)**4+2*(1+delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    sd = numerator*((denominator+1e-26)**-1)
    sd = sd.real

    vp2dt2 = vp**2*dt**2

    # Background wavefield
    # 10.1190/geo2022-0292.1 EQ(A-7)
    ani_laplace_p0 = vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z-2*eps*vp2dt2*sin20*dpdxdz
    pnext = 2*p1-p2 + ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = _laplacian(sp1, h, kernelx)
    nabla_z_s = _laplacian(sp1, h, kernely)

    dpdx_s = gradient(sp1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz_s = gradient(dpdx_s, h, kernely_nc)

    spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x_s + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z_s-2*eps*vp2dt2*sin20*dpdxdz_s+m*ani_laplace_p0

    # with torch.no_grad():
    pnext = restore_boundaries(pnext, h_bd)
    spnext = restore_boundaries(spnext, sh_bd)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1, spnext, sp1

def _time_step_backward_multiple(*args, **kwargs):
    vp, eps, delta, theta, m = args[0:5]
    p1, p2, sp1, sp2 = args[5:9]
    dt, h, b = args[9:12]
    h_bd, _, sh_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    # from degree to radian
    theta = torch.deg2rad(theta)
    sin0 = torch.sin(theta)
    cos0 = torch.cos(theta)
    sin20 = torch.sin(2*theta)

    nabla_x = _laplacian(p1, h, kernelx)
    nabla_z = _laplacian(p1, h, kernely)

    dpdx = gradient(p1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz = gradient(dpdx, h, kernely_nc)

    # 10.1190/geo2022-0292.1 EQ(A-5)
    shape = p1.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(p1.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(p1.device)
    k_x, k_z = torch.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    denominator = (1+2*eps)*(k_x*cos0-k_z*sin0)**4+(k_x*sin0+k_z*cos0)**4+2*(1+delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    sd = numerator*((denominator+1e-26)**-1)
    sd = sd.real

    vp2dt2 = vp**2*dt**2

    # Background wavefield
    # 10.1190/geo2022-0292.1 EQ(A-7)
    ani_laplace_p0 = vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z-2*eps*vp2dt2*sin20*dpdxdz
    pnext = 2*p1-p2 + ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = _laplacian(sp1, h, kernelx)
    nabla_z_s = _laplacian(sp1, h, kernely)

    dpdx_s = gradient(sp1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz_s = gradient(dpdx_s, h, kernely_nc)
    
    spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x_s + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z_s-2*eps*vp2dt2*sin20*dpdxdz_s+m*ani_laplace_p0

    # with torch.no_grad():
    pnext = restore_boundaries(pnext, h_bd, multiple=True)
    spnext = restore_boundaries(spnext, sh_bd, multiple=True)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1, spnext, sp1
