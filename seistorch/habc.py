import torch
import numpy as np

def bound_mask(nz, nx, w, dev, batchsize=1, return_idx=False):

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

    if not return_idx:
        return top, bottom, left, right
    else:
        if batchsize>1:
            tm = top.repeat(batchsize, 1, 1)==1
            bm = bottom.repeat(batchsize, 1, 1)==1
            lm = left.repeat(batchsize, 1, 1)==1
            rm = right.repeat(batchsize, 1, 1)==1
        return tm, bm, lm, rm

def generate_habc_coefficients_2d(domain_shape, 
                                  N=50, 
                                  multiple=False, 
                                  device='cpu'):

    nz, nx = domain_shape

    d = torch.zeros(nz, nx, device=device)

    d_vals = torch.linspace(0.0, N, N, device=device)/N
    d_vals = torch.flip(d_vals, [0])
    
    tm, bm, lm, rm = bound_mask(*domain_shape, N, dev=device)

    if N > 0:
        # Top
        idx = tm==1
        d[:N,:][idx] = (d_vals.repeat(nx, 1).transpose(0, 1)*tm)[idx]

        # Bottom
        idx = bm==1 # Mask for equation left
        d[-N:,:][idx] = (torch.flip(d_vals, [0]).repeat(nx, 1).transpose(0, 1)*bm)[idx]

        # Left
        idx = lm==1
        d[:, :N][idx] = (d_vals.repeat(nz, 1)*lm)[idx]

        # Right boundary
        idx = rm==1
        d[:, -N:][idx] = (torch.flip(d_vals, [0]).repeat(nz, 1)*rm)[idx]

    return d
