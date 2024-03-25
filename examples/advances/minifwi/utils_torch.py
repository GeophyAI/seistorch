import torch, tqdm
import numpy as np
import matplotlib.pyplot as plt

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
    return torch.nn.functional.conv2d(u, kernel, padding=1) / (h ** 2)

def step(u_pre, u_now, c=1.5, dt=0.001, h=10/1000.):
    u_next = 2 * u_now - u_pre + (c * dt) ** 2 * laplace(u_now, h)
    return u_next

def forward(wave, c, src_list, domain, dt, h, dev, recz=0):
    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)
    u_pre = torch.zeros(nshots, 1, *domain).to(dev)
    u_now = torch.zeros(nshots, 1, *domain).to(dev)
    rec = torch.zeros(nshots, nt, nx).to(dev)
    c = c.unsqueeze(0)
    shots = torch.arange(nshots).to(dev)
    srcx, srcz = zip(*src_list)

    for it in range(nt):
        # u_now = u_now.clone()

        u_now[shots, :, srcz, srcx] += wave[it]

        # for ishot in range(nshots):
        #     sx, sz = src_list[ishot]
        #     u_now[ishot, 0, sz, sx] += wave[it]
        u_next = step(u_pre, u_now, c, dt, h)
        u_pre = u_now
        u_now = u_next
        rec[:,it, :] = u_now[:, 0, recz, :]
    return rec