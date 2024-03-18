import torch
from torch.nn.functional import conv2d
from .utils import restore_boundaries
from .acoustic_habc import habc, generate_convolution_kernel, _laplacian

spatial_order = 2
device = "cuda"
kernel = generate_convolution_kernel(spatial_order).unsqueeze(0).unsqueeze(0).to(device)
padding = kernel.shape[-1]//2

def _time_step(*args, **kwargs):
    
    vp, m = args[0:2]
    h1, h2, sh1, sh2 = args[2:6]
    dt, h, b = args[6:9]
    habc_masks = kwargs['habcs']

    # background wavefield
    vp2_nabla_p0 = vp**2*_laplacian(h1, h)
    p0 = 2*h1-h2 + vp2_nabla_p0*dt**2

    # scatter wavefield
    vp2_nabla_sh0 = vp**2*_laplacian(sh1, h)
    sh0 = 2*sh1-sh2 + vp2_nabla_sh0*dt**2 + m*vp2_nabla_p0*dt**2

    # HABC
    p0 = habc(p0, h1, h2, vp, b, dt, h, maskidx = habc_masks)
    sh0 = habc(sh0, sh1, sh2, vp, b, dt, h, maskidx = habc_masks)

    return p0, h1, sh0, sh1


def _time_step_backward(*args, **kwargs):
    bwidth=50
    N=1
    vp, m = args[0:2]
    h1, h2, sh1, sh2 = args[2:6]
    dt, h, b = args[6:9]
    h_bd, _, sh_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    vp = vp.unsqueeze(0)
    b = b.unsqueeze(0)
    
    # background wavefield
    vp2_nabla_p0 = vp**2*_laplacian(h1, h)
    p0 = 2*h1-h2 + vp2_nabla_p0*dt**2

    # scatter wavefield
    vp2_nabla_sh0 = vp**2*_laplacian(sh1, h)
    sh0 = 2*sh1-sh2 + vp2_nabla_sh0*dt**2 + m*vp2_nabla_p0*dt**2

    with torch.no_grad():
        p0 = restore_boundaries(p0, h_bd)
        sh0 = restore_boundaries(sh0, sh_bd)

    p0 = src_func(p0, src_values, 1)

    return p0, h1, sh0, sh1

def _time_step_backward_multiple(*args, **kwargs):

    vp, m = args[0:2]
    h1, h2, sh1, sh2 = args[2:6]
    dt, h, b = args[6:9]
    h_bd, _, sh_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    vp = vp.unsqueeze(0)
    b = b.unsqueeze(0)
    
    # background wavefield
    vp2_nabla_p0 = vp**2*_laplacian(h1, h)
    p0 = 2*h1-h2 + vp2_nabla_p0*dt**2

    # scatter wavefield
    vp2_nabla_sh0 = vp**2*_laplacian(sh1, h)
    sh0 = 2*sh1-sh2 + vp2_nabla_sh0*dt**2 + m*vp2_nabla_p0*dt**2

    with torch.no_grad():
        p0 = restore_boundaries(p0, h_bd, multiple=True)
        sh0 = restore_boundaries(sh0, sh_bd, multiple=True)

    p0 = src_func(p0, src_values, 1)

    return p0, h1, sh0, sh1
