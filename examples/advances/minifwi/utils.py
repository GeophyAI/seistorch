import torch, jax
import jax.numpy as jnp

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

class Propagator:

    def __init__(self, backend="torch"):
        assert backend in ["torch", "jax"]
        self.backend = backend
        # self.laplace = eval(f"laplace_{backend}")
        self.laplace = getattr(self, f"laplace_{backend}")
        self.addSources = eval(f"addSources_{backend}")
        pass

    @property
    def device(self):
        if self.backend == "torch":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.backend == "jax":
            return jax.devices()[0]
    
    @property
    def kernel(self):
        kernel = [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]
        if self.backend == "torch":
            return torch.tensor(kernel).to(self.device)
        elif self.backend == "jax":
            return jnp.array(kernel)

    def step(self, u_pre, u_now, c, dt, h):
        print(u_pre.devices(), u_now.devices(), c.devices())
        u_next = 2 * u_now - u_pre + (c * dt) ** 2 * self.laplace(u_now, h)
        return u_next
    
    def zeros(self, size, dev):
        if self.backend == "torch":
            return torch.zeros(size).to(dev)
        elif self.backend == "jax":
            return jnp.zeros(size)

    def forward(self, wave, c, src_list, domain, dt, h, dev, recz=0):
        nt = wave.shape[0]
        nz, nx = domain
        nshots = len(src_list)
        u_pre = self.zeros((nshots, 1, *domain), dev)
        u_now = self.zeros((nshots, 1, *domain), dev)
        rec = self.zeros((nshots, nt, nx), dev)
        # c = c.unsqueeze(0)
        for it in range(nt):
            # u_now = u_now.clone()
            for ishot in range(nshots):
                sx, sz = src_list[ishot]
                u_now = self.addSources(u_now, ishot, sz, sx, wave[it])
                # u_now[ishot, 0, sz, sx] += wave[it]
            u_next = self.step(u_pre, u_now, c, dt, h)
            u_pre = u_now
            u_now = u_next
            # rec[:,it, :] = u_now[:, 0, recz, :]
        return rec
    
    @jax.jit
    def laplace_jax(self, u, h):
        return jax.lax.conv_general_dilated(u, self.kernel, (1,1), 'SAME', (1,1), (1,1)) / (h ** 2)

    def laplace_torch(self, u, h):
        return torch.nn.functional.conv2d(u, self.kernel, padding=1) / (h ** 2)
            

def ricker(t, f):
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return r

        
@jax.jit
def addSources_jax(u, ishot, sz, sx, s):
    # Not inplace
    u = u.at[ishot, :, sz, sx].add(s)
    return u

def addSources_torch(u, ishot, sz, sx, s):
    # Inplace
    u[ishot, 0, sz, sx] += s
    return u



