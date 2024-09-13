import torch, tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import conv3d

def imshow(data, vmin=None, vmax=None, cmap=None, figsize=(10, 10), savepath=None):
    plt.figure(figsize=figsize)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar()
    if savepath:
        plt.savefig(savepath)
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
# configure 3D kernel

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
    kernel = torch.zeros((kernel_size, kernel_size, kernel_size))
    center = spatial_order // 2
    kernel[center, center+1:, center] = constant
    kernel[center, 0:center, center] = constant.flip(0)

    kernel[center+1:, center, center] = constant
    kernel[0:center, center, center] = constant.flip(0)

    kernel = torch.transpose(kernel, 1, 2)

    kernel[center, center+1:, center] = constant
    kernel[center, 0:center, center] = constant.flip(0)

    kernel[center, center, center] = -2*3*torch.sum(constant)

    return kernel

spatial_order = 2
device = "cuda"
kernel = generate_convolution_kernel(spatial_order).unsqueeze(0).unsqueeze(0).to(device)
padding = kernel.shape[-1]//2

def ricker(t, f=10):
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return torch.from_numpy(r).float().to(device)

def laplace(u, h):
    return conv3d(u, kernel, padding=padding) / (h ** 2)

def step(u_pre, u_now, c=1.5, dt=0.001, h=10./1000., b=None):
    _laplace_u = laplace(u_now, h)
    a = (dt**-2 + b * dt**-1)**(-1)
    u_next = a*(2. / dt**2 * u_now - (dt**-2-b*dt**-1)*u_pre + c**2*_laplace_u)
    return u_next

def forward(wave, c, b, src_list, rec_list, domain, dt, h, dev, recz=0, pmln=50):
    nt = wave.shape[0]
    nz, ny, nx = domain
    nshots = len(src_list)
    u_pre = torch.zeros(nshots, *domain, device=dev)
    u_now = torch.zeros(nshots, *domain, device=dev)
    rec = torch.zeros(nshots, nt, rec_list.shape[-1], device=dev)
    b = b.unsqueeze(0).to(dev)
    c = c.unsqueeze(0)
    shots = torch.arange(nshots, device=dev)
    srcx, srcy, srcz = zip(*src_list)
    recx, recy, recz = rec_list[0,0], rec_list[0,1], rec_list[0,2]
    h = torch.Tensor([h]).to(dev)
    dt = torch.Tensor([dt]).to(dev)

    source_mask = torch.zeros_like(u_now)
    source_mask[shots, srcz, srcy, srcx] = 1

    for it in range(nt):
        u_now += source_mask * wave[it]
        u_next = step(u_pre, u_now, c, dt, h, b)
        u_pre, u_now = u_now, u_next
        rec[:, it, :] = u_now[:, recz, recy, recx]

    return rec

def generate_pml_coefficients_3d(domain_shape, N=50, B=100., multiple=False):
    nz, ny, nx = domain_shape
    # Cosine coefficients for pml
    idx = (torch.ones(N + 1) * (N+1)  - torch.linspace(0.0, (N+1), N + 1))/(2*(N+1))
    b_vals = torch.cos(torch.pi*idx)
    b_vals = torch.ones_like(b_vals) * B * (torch.ones_like(b_vals) - b_vals)

    b_x = torch.zeros((nz, ny, nx))
    b_y = torch.zeros((nz, ny, nx))
    b_z = torch.zeros((nz, ny, nx))

    b_x[:,0:N+1,:] = b_vals.repeat(nx, 1).transpose(0, 1)
    b_x[:,(ny - N - 1):ny,:] = torch.flip(b_vals, [0]).repeat(nx, 1).transpose(0, 1)

    b_y[:,:,0:N + 1] = b_vals.repeat(ny, 1)
    b_y[:,:,(nx - N - 1):nx] = torch.flip(b_vals, [0]).repeat(ny, 1)

    b_z[0:N + 1, :, :] = b_vals.view(-1, 1, 1).repeat(1, ny, nx)
    b_z[(nz - N - 1):nz + 1, :, :] = torch.flip(b_vals, [0]).view(-1, 1, 1).repeat(1, ny, nx)

    return torch.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2)