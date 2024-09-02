import torch, tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../../')
from seistorch.equations2d.acoustic_habc import habc

def imshow(data, vmin=None, vmax=None, cmap=None, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.show()

def generate_mesh(mshape, dh=1):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors_for_meshgrid = []
    for size in mshape:
        tensors_for_meshgrid.append(torch.linspace(-1, 1, steps=size))
        # tensors_for_meshgrid.append(torch.linspace(0, size*dh/1000, steps=size))
    mgrid = torch.stack(torch.meshgrid(*tensors_for_meshgrid, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, len(mshape))
    return mgrid

def show_gathers(rec, size=3, figsize=(8, 5)):
    randno = np.random.randint(0, rec.shape[0], size=size)
    fig,axes=plt.subplots(1, randno.shape[0], figsize=figsize)
    for i, ax in enumerate(axes):
        vmin,vmax=np.percentile(rec[i], [1, 99])
        kwargs=dict(vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.imshow(rec[randno[i]], **kwargs)
        ax.set_title(f"shot {randno[i]}")
    plt.tight_layout()
    plt.show()

def showgeom(vel, src_loc, rec_loc, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(vel, vmin=vel.min(), vmax=vel.max(), cmap="seismic", aspect="auto")
    plt.colorbar()
    plt.scatter(*zip(*src_loc), c="r", marker="v", s=100, label="src")
    plt.scatter(*zip(*rec_loc), c="b", marker="^", s=10, label="rec")
    plt.legend()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configure
kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(device)

def ricker(t, f=10):
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return torch.from_numpy(r).float().to(device)

def laplace(u, h):
    u = u.unsqueeze(1)
    return torch.nn.functional.conv2d(u, kernel, padding=1).squeeze(1) / (h ** 2)

def step(h1, h2, sh1, sh2, m, vp=1.5, dt=0.001, h=10./1000., b=None, habc_masks=None, bwidth=20):

    a1 = (2-b**2*dt**2)/(1+b*dt)
    a2 = (1-b*dt)/(1+b*dt)
    a3 = vp**2*dt**2/(1+b*dt)
    # background wavefield
    vp2_nabla_p0 = vp**2*laplace(h1, h)
    # p0 = 2*h1-h2 + vp2_nabla_p0*dt**2
    p0 = a1*h1 - a2*h2 + a3*laplace(h1, h)

    # scatter wavefield
    # vp2_nabla_sh0 = vp**2*laplace(sh1, h)
    # sh0 = 2*sh1-sh2 + vp2_nabla_sh0*dt**2 + m*vp2_nabla_p0*dt**2
    sh0 = a1*sh1 - a2*sh2 + a3*laplace(sh1, h) + m*vp2_nabla_p0*dt**2
    # HABC
    # p0 = habc(p0, h1, h2, vp, b, dt, h, w=bwidth, maskidx=habc_masks)
    # sh0 = habc(sh0, sh1, sh2, vp, b, dt, h, w=bwidth, maskidx=habc_masks)

    return p0, sh0

def forward(wave, m, c, b, src_list, domain, dt, h, dev, recz=0, bwidth=50, habc_masks=None):

    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)
    u_pre = torch.zeros(nshots, *domain, device=dev)
    u_now = torch.zeros(nshots, *domain, device=dev)
    su_pre = torch.zeros(nshots, *domain, device=dev)
    su_now = torch.zeros(nshots, *domain, device=dev)
    rec = torch.zeros(nshots, nt, nx-2*bwidth, device=dev)

    shots = torch.arange(nshots, device=dev)
    srcx, srcz = zip(*src_list)

    h = torch.Tensor([h]).to(dev)
    dt = torch.Tensor([dt]).to(dev)

    source_mask = torch.zeros_like(u_now)
    source_mask[shots, srcz, srcx] = 1

    forward_kwargs = dict(dt=dt, h=h, b=b, bwidth=bwidth, habc_masks=habc_masks)

    for it in range(nt):
        u_now += source_mask * wave[it]
        u_next, su_next = step(u_now, u_pre, su_now, su_pre, m, c, **forward_kwargs)
        u_pre, u_now = u_now, u_next
        su_pre, su_now = su_now, su_next
        rec[:, it, :] = su_now[:, recz, bwidth:-bwidth]
    return rec

def generate_pml_coefficients_2d(domain_shape, N=50, B=50., multiple=False):
    
    Nx, Ny = domain_shape
    xx = torch.linspace(0, 1, N+1)

    d_vals = B*(1-torch.cos(torch.pi*xx/2))
    d_vals = torch.flip(d_vals, [0])

    d_x = torch.zeros(Ny, Nx)
    d_y = torch.zeros(Ny, Nx)
    
    if N > 0:
        d_x[0:N + 1, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
        d_x[(Ny - N - 1):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
        if not multiple:
            d_y[:, 0:N + 1] = d_vals.repeat(Ny, 1)
        d_y[:, (Nx - N - 1):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

    _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)
    _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, multiple)

    return _d

def _corners(domain_shape, abs_N, d, dx, dy, multiple=False):
    Nx, Ny = domain_shape
    for j in range(Ny):
        for i in range(Nx):
            # Left-Top
            if not multiple:
                if i < abs_N+1 and j< abs_N+1:
                    if i < j: d[i,j] = dy[i,j]
                    else: d[i,j] = dx[i,j]
            # Left-Bottom
            if i > (Nx-abs_N-2) and j < abs_N+1:
                if i + j < Nx: d[i,j] = dx[i,j]
                else: d[i,j] = dy[i,j]
            # Right-Bottom
            if i > (Nx-abs_N-2) and j > (Ny-abs_N-2):
                if i - j > Nx-Ny: d[i,j] = dy[i,j]
                else: d[i,j] = dx[i,j]
            # Right-Top
            if not multiple:
                if i < abs_N+1 and j> (Ny-abs_N-2):
                    if i + j < Ny: d[i,j] = dy[i,j]
                    else: d[i,j] = dx[i,j]

    return d