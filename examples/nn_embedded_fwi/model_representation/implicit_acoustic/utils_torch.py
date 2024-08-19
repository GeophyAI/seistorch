import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle   

def imshow(data, vmin=None, vmax=None, cmap=None, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.show()

def show_gathers(rec, size=3, figsize=(8, 5), savepath=''):
    randno = np.random.randint(0, rec.shape[0], size=size)
    fig,axes=plt.subplots(1, randno.shape[0], figsize=figsize)

    if size==1:
        axes=[axes]

    for i, ax in enumerate(axes):
        vmin,vmax=np.percentile(rec[i], [1, 99])
        kwargs=dict(vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.imshow(rec[randno[i]], **kwargs)
        ax.set_title(f"shot {randno[i]}")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()

def showgeom(vel, src_loc, rec_loc, figsize=(10, 10), savepath=''):
    plt.figure(figsize=figsize)
    plt.imshow(vel, vmin=vel.min(), vmax=vel.max(), cmap="seismic", aspect="auto")
    plt.colorbar()
    plt.scatter(*zip(*src_loc), c="r", marker="v", s=100, label="src")
    plt.scatter(*zip(*rec_loc), c="b", marker="^", s=10, label="rec")
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configure
kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(device)

def ricker(t, f=10):
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return torch.from_numpy(r).float().to(device)

def laplace(u, h):
    return torch.nn.functional.conv2d(u, kernel, padding=1) / (h ** 2)

def step(u_pre, u_now, c=1.5, dt=0.001, h=10./1000., b=None):

    # With boundary condition
    u_next = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * u_now - torch.mul((dt**-2 - b * dt**-1), u_pre)
                + torch.mul(c.pow(2), laplace(u_now, h)))
                )
    # u_next = ((dt**-2 + b * dt**-1).pow(-1) *
    #             (2 / dt**2 * u_now - (dt**-2 - b * dt**-1) * u_pre
    #             + c.pow(2) * laplace(u_now, h)))
    # Without boundary condition
    # u_next = 2 * u_now - u_pre + (c * dt) ** 2 * laplace(u_now, h)
    return u_next

def forward(wave, c, b, src_list, domain, dt, h, dev, recz=0, pmln=50):
    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)
    u_pre = torch.zeros(nshots, 1, *domain).to(dev)
    u_now = torch.zeros(nshots, 1, *domain).to(dev)
    rec = torch.zeros(nshots, nt, nx-2*pmln).to(dev)
    b = b.unsqueeze(0).to(dev)
    c = c.unsqueeze(0)
    shots = torch.arange(nshots).to(dev)
    srcx, srcz = zip(*src_list)

    h = torch.Tensor([h]).to(dev)
    dt = torch.Tensor([dt]).to(dev)

    source_mask = torch.zeros_like(u_now)
    source_mask[shots, :, srcz, srcx] = 1

    for it in range(nt):
        u_now += source_mask * wave[it]
        u_next = step(u_pre, u_now, c, dt, h, b)
        u_pre, u_now = u_now, u_next
        rec[:,it, :] = u_now[:, 0, recz, pmln:-pmln]
    return rec

def generate_pml_coefficients_2d(domain_shape, N=50, B=100., multiple=False):
    Nx, Ny = domain_shape

    R = 10**(-((np.log10(N)-1)/np.log10(2))-3)
    #d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
    R = 1e-6; order = 2; cp = 1000.
    d0 = (1.5*cp/N)*np.log10(R**-1)
    d_vals = d0 * torch.linspace(0.0, 1.0, N + 1) ** order
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

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)