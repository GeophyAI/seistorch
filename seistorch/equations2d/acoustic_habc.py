import torch
import numpy as np
from torch.nn.functional import conv2d
from .utils import restore_boundaries

def even_intergrid(n: int):
    """Calculate the coefficients of taylar expansion of even intergrid

    Args:
        n (int): The spatial order of the laplace operator.
    """
    def fact(n):
        if n == 0:
            return 1
        else:
            return n * fact(n-1)
    
    MAX = n // 2
    matrix = torch.zeros((MAX, MAX))
    constant_one = torch.zeros((MAX,))
    for i in range(MAX):
        for j in range(MAX):
            matrix[i, j] = (j + 1)**(2*(i+1) - 2)
        if i != ((2+1)//2 - 1):
            constant_one[i] = 0
        else:
            constant_one[i] = (1**(2*(i+1)-2) * fact(2*(i+1)-2))

    C = torch.linalg.solve(matrix, constant_one.unsqueeze(1))
    constant = C.squeeze() / torch.tensor([(i+1)**2 for i in range(MAX)], dtype=torch.float32)
    return constant

def generate_convolution_kernel(spatial_order):
    """Generate convolution kernel

    Args:
        n (int): The order of the taylor expansion

    Returns:
        _type_: Tensor, the convolution kernel
    """
    constant = even_intergrid(spatial_order)
    kernel_size = spatial_order + 1
    kernel = torch.zeros((kernel_size, kernel_size))
    center = spatial_order // 2
    kernel[center, center+1:] = constant
    kernel[center, 0:center] = constant.flip(0)

    kernel[center+1:, center] = constant
    kernel[0:center, center] = constant.flip(0)

    kernel[center, center] = -2*2*torch.sum(constant)

    return kernel

spatial_order = 2
device = "cuda"
kernel = generate_convolution_kernel(spatial_order).unsqueeze(0).unsqueeze(0).to(device)
padding = kernel.shape[-1]//2

def _laplacian(y, h):
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

def cutb(d, w=30, n=1):
    if d.ndim == 3:
        return d[:, :w+n, :]
    else:
        return d[:w+n, :]

def _habc(u_next, u_now, u_pre, c, b, dt, dh, w=30):
    cut = w
    # w += 2
    u_next = u_next#[:, :w, :]
    u_now = u_now#[:, :w, :]
    u_pre = u_pre#[:, :w, :]
    c = c#[:w, :]
    b = b#[:w, :]

    lam = 2*c*dt/dh
    mu = c**2*dt**2/(dh**2)

    # Top
    t1 = (2 - lam - mu) * u_now
    t2 = (lam + 2 * mu) * torch.roll(u_now, shifts=-1, dims=2)
    t3 = -mu * torch.roll(u_now, shifts=-2, dims=2)
    t4 = (lam - 1) * u_pre
    t5 = -lam * torch.roll(u_pre, shifts=-1, dims=2)
    u_one = t1 + t2 + t3 + t4 + t5

    hb_next = (u_one*b + (1-b) * u_next)

    return hb_next[:, :, :cut, :]

def permute(d):
    if d.ndim == 3:
        return d.permute(0, 2, 1)
    elif d.ndim == 2:
        return d.T
    
def rot90(d, k=1):
    if d.ndim == 3:
        return torch.rot90(d, k=k, dims=(1, 2))
    elif d.ndim == 2:
        return torch.rot90(d, k=k, dims=(0, 1))
    elif d.ndim == 4 :
        return torch.rot90(d, k=k, dims=(2, 3))
    
def flipud(d):
    if d.ndim == 3:
        return torch.flip(d, dims=[1])
    elif d.ndim == 2:
        return torch.flip(d, dims=[0])
    
def identity(d):
    return d

def stack(*args):
    return torch.stack(args, dim=0)

def bound_mask(nz, nx, w, dev):

    top = torch.ones(w, nx, device=dev)

    indices = np.tril_indices(w, k=-1)

    top[indices] = 0.0
    top *= torch.fliplr(top)
    bottom = torch.flipud(top)

    left = torch.ones(nz, w, device=dev)
    indices = np.triu_indices(w, k=1)
    left[indices] = 0.0
    left *= torch.flipud(left)
    right = torch.fliplr(left)

    return top, bottom, left, right

def habc(y, h1, h2, vel, coes, dt, h, w=30, maskidx=None):


    otherargs = [dt, h]
    # Calculate weighted one/two-wave-wavefield
    tbargs = [stack(cutb(array), cutb(flipud(array))) for array in [y, h1, h2, vel.unsqueeze(0), coes.unsqueeze(0)]]+otherargs
    lrargs = [stack(cutb(rot90(array, -1)), cutb(rot90(array, 1))) for array in [y, h1, h2, vel.unsqueeze(0), coes.unsqueeze(0)]]+otherargs

    top, bottom = torch.split(_habc(*tbargs), 1, dim=0)
    left, right = torch.split(_habc(*lrargs), 1, dim=0)
    tmidx, bmidx, lmidx, rmidx = maskidx

    multiple = tmidx is None

    """Rotate"""
    y_top = top.squeeze()
    y_bottom = torch.flip(bottom, dims=[2]).squeeze()
    y_left = rot90(left).squeeze()
    y_right = rot90(right, -1).squeeze()

    # Top
    # print(y_top.shape, y.shape, tmidx.shape)
    if not multiple:
        idxl = tmidx[None, :, :] if y.ndim != tmidx.ndim else tmidx
        y[:,:w,:][idxl] = y_top[tmidx]

    # Bottom
    idxl = bmidx[None, :, :] if y.ndim != bmidx.ndim else bmidx
    y[:,-w:,:][idxl] = y_bottom[bmidx]

    # Left
    idxl = lmidx[None, :, :] if y.ndim != lmidx.ndim else lmidx
    y[:, :, :w][idxl] = y_left[lmidx]

    # Right boundary
    idxl = rmidx[None, :, :] if y.ndim != rmidx.ndim else rmidx
    y[:, :, -w:][idxl] = y_right[rmidx]

    dix, diz = np.diag_indices(w)

    if not multiple:
        # Top-Left corner
        y[:, dix, diz] = 0.5*y_top[..., dix, diz]\
                    + 0.5*y_left[..., dix, diz]
        
        # Top-Right corner
        y[:, dix, -w+diz] = 0.5*y_top[..., dix, -w+diz]\
                        + 0.5*y_right[..., dix, -w+diz]
    
    # Bottom-Left corner
    y[:, -w+dix, diz] = 0.5*y_bottom[..., dix, diz]\
                      + 0.5*y_left[..., -w+dix, diz]
    
    # Bottom-Right corner
    y[:, -w+dix, -w+diz] = 0.5*y_bottom[..., dix, -w+diz]\
                         + 0.5*y_right[..., -w+dix, -w+diz]

    return y

def _time_step(*args, **kwargs):

    c = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    habc_masks = kwargs['habcs']
    
    # HABC
    y = torch.mul((dt**-2).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2 ), h2)
                + torch.mul(c.pow(2), _laplacian(h1, h)))
                )
    
    y = habc(y, h1, h2, c, b, dt, h, maskidx = habc_masks)

    return y, h1

def _time_step_backward(*args, **kwargs):

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
    
    with torch.no_grad():
        y = restore_boundaries(y, h_bd)
    
    y = src_func(y, src_values, 1)

    return y, h1