import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.signal import convolve2d as conv2d
from jax import vmap
from jax import lax
import torch
from jax import jvp, grad
import copy
from functools import partial

# Ricker wavelet
def ricker(t, f=10):
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return jnp.array(r)

# @jax.jit
def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))

# Laplace operator
# @jax.jit
def laplace(u, h):
    kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 3x3 kernel
    return batch_convolve2d(u, kernel) / (h ** 2)

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

# time step forward
# @partial(jax.jit, static_argnames=['pmln'])
# def step(u_pre, u_now, c=1.5, dt=0.001, h=10./1000., b=None, last_step=False, pmln=50, *args, **kwargs):
#     _laplace_u = laplace(u_now, h)
#     a = (dt**-2 + b * dt**-1)**(-1)
#     u_next = a*(2. / dt**2 * u_now - (dt**-2-b*dt**-1)*u_pre + c**2*_laplace_u)
#     return u_next

def step(*args, **kwargs):
    u_pre, u_now, c, dt, h, b, last_step, pmln = args
    _laplace_u = laplace(u_now, h)
    a = (dt**-2 + b * dt**-1)**(-1)
    u_next = a*(2. / dt**2 * u_now - (dt**-2-b*dt**-1)*u_pre + c**2*_laplace_u)
    return u_next

# @partial(jax.jit, static_argnames=['pmln'])
def step_fwd(u_pre, u_now, c=1.5, dt=0.001, h=10./1000., b=None, last_step=False, pmln=50):
    u_next = step(u_pre, u_now, c, dt, h, b, last_step, pmln)
    # top = u_next[:, :, :pmln, :]
    # bottom = u_next[:, :, -pmln:, :]
    # left = u_next[:, :, :, :pmln]
    # right = u_next[:, :, :, -pmln:]

    return u_next, ((u_pre, u_now, c, dt, h, b, last_step, pmln), (None, None, None, None))

    # if last_step:
    #     return u_next, (u_pre, u_now, c, dt, h, b)
    # else:
    #     top = u_next[:, :, :pmln, :]
    #     bottom = u_next[:, :, -pmln:, :]
    #     left = u_next[:, :, :, :pmln]
    #     right = u_next[:, :, :, -pmln:]
    #     return u_next, (top, bottom, left, right, c, dt, h, b)

# @partial(jax.jit, static_argnums=(0,))
def step_bwd(res, g):

    _, vjp_fun = jax.vjp(step, *res[0])
    grads = vjp_fun(g)

    return grads

# forward modeling
def forward(wave, c, b, src_list, domain, dt, h, recz=0, pmln=50):

    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)
    u_pre = jnp.zeros((nshots, 1, *domain))
    u_now = jnp.zeros((nshots, 1, *domain))
    rec = jnp.zeros((nshots, nt, nx-2*pmln))
    b = b
    c = c

    shots = jnp.arange(0, nshots, 1)
    srcx, srcz = zip(*src_list)
    source_mask = jnp.zeros((nshots, 1, *domain))
    source_mask = source_mask.at[shots, 0, srcz, srcx].set(1)

    # _step = jax.custom_vjp(step)
    # _step.defvjp(step_fwd, step_bwd)

    _step = step

    def step_fn(carry, it):
        u_pre, u_now, rec = carry
        source = wave[it] * source_mask
        u_now = u_now + source
        u_next = _step(u_pre, u_now, c, dt, h, b, it==nt-1, pmln)
        rec = rec.at[:, it, :].set(u_now[:, 0, recz, pmln:-pmln])
        return (u_now, u_next, rec), None

    initial_carry = (u_pre, u_now, rec)

    final_carry, _ = lax.scan(step_fn, initial_carry, jnp.arange(nt))

    _, _, rec_final = final_carry

    return rec_final

# Coefficients of PML
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