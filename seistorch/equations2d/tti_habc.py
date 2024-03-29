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
    operator = h ** (-1) * kernel.to(y.device)
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=padding).squeeze(1)

def _time_step(*args, **kwargs):

    vp, eps, delta = args[0:3]
    p1, p2 = args[3:5]
    dt, h, b = args[5:8]
    habc_masks = kwargs['habcs']

    theta = -45. # degree
    # from degree to radian
    theta = torch.Tensor([theta*np.pi/180]).float().to(vp.device)
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

    # 10.1190/geo2022-0292.1 EQ(A-7)
    pnext = 2*p1-p2 + vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z-2*eps*vp2dt2*sin20*dpdxdz

    # apply HABC
    pnext = habc(pnext, p1, p2, vp, b, dt, h, maskidx = habc_masks)

    return pnext, p1

def _time_step_backward(*args, **kwargs):

    vp, eps, delta = args[0:3]
    p1, p2 = args[3:5]
    dt, h, b = args[5:8]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    theta = -45. # degree
    # from degree to radian
    theta = torch.Tensor([theta*np.pi/180]).float().to(vp.device)
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

    # 10.1190/geo2022-0292.1 EQ(A-7)
    pnext = 2*p1-p2 + vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z-2*eps*vp2dt2*sin20*dpdxdz

    with torch.no_grad():
        pnext = restore_boundaries(pnext, h_bd)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1

def _time_step_backward_multiple(*args, **kwargs):

    vp, eps, delta = args[0:3]
    p1, p2 = args[3:5]
    dt, h, b = args[5:8]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    theta = -45. # degree
    # from degree to radian
    theta = torch.Tensor([theta*np.pi/180]).float().to(vp.device)
    sin0 = torch.sin(theta)
    cos0 = torch.cos(theta)
    sin20 = torch.sin(2*theta)

    nabla_x = _laplacian(p1, h, kernelx)
    nabla_z = _laplacian(p1, h, kernely)

    # Method 1: solver in time domain
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

    # 10.1190/geo2022-0292.1 EQ(A-7)
    pnext = 2*p1-p2 + vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z-2*eps*vp2dt2*sin20*dpdxdz
    with torch.no_grad():
        pnext = restore_boundaries(pnext, h_bd, multiple=True)

    pnext = src_func(pnext, src_values, 1)

    return pnext, p1
