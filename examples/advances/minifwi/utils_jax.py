import jax, tqdm
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import lax, ops
from functools import partial

# Check if GPU is available
device = jax.devices("gpu" if jax.devices("gpu") else "cpu")[0]

# Configure
kernel = jnp.array([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]])

# repeat the kernel for each channel
# kernel = jnp.repeat(kernel, 1, axis=0)

def ricker(t, f):
    r = (1 - 2 * (jnp.pi * f * t) ** 2) * jnp.exp(-(jnp.pi * f * t) ** 2)
    return r

@jax.jit
def laplace(u, h):
    return lax.conv_general_dilated(u, kernel, (1,1), 'SAME', (1,1), (1,1)) / (h ** 2)
    # return lax.conv(u, kernel, (1,1), 'SAME') / (h ** 2)

@jax.jit
def step(u_pre, u_now, c, dt, h):
    u_next = 2 * u_now - u_pre + (c * dt) ** 2 * laplace(u_now, h)
    return u_next

@jax.jit
def addsources(u_now, shots, sz, sx, s):
    u_now = u_now.at[shots, :, sz, sx].add(s)
    return u_now

# @jax.jit
# def record(rec, it, u_now, shots, recz):
#     rec = rec.at[:, it, :].set(u_now[:, 0, recz, :])
#     return rec

@jax.jit
def record(u_now, recz):
    return u_now[:, 0, recz, :]


def forward(wave, c, src_list, domain, dt, h, recz):
    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)
    u_pre = jnp.zeros((nshots, 1, nz, nx))
    u_now = jnp.zeros((nshots, 1, nz, nx))
    rec = jnp.zeros((nshots, nt, nx))
    shots = jnp.arange(nshots)
    srcx, srcz = zip(*src_list)
    def scan_fn(carry, wave_t):
        u_pre, u_now, rec = carry
        u_now = addsources(u_now, shots, srcz, srcx, wave_t)
        u_next = step(u_pre, u_now, c, dt, h)
        return u_now, u_next, record(u_now, recz)
    
    _, _, rec = jax.lax.scan(scan_fn, (u_pre, u_now, rec), wave)

    return rec

# def forward(wave, c, src_list, domain, dt, h, recz):
#     nt = wave.shape[0]
#     nz, nx = domain
#     nshots = len(src_list)
#     u_pre = jnp.zeros((nshots, 1, nz, nx))
#     u_now = jnp.zeros((nshots, 1, nz, nx))
#     rec = jnp.zeros((nshots, nt, nx))
#     c = jnp.expand_dims(c, axis=0)
#     shots = jnp.arange(nshots)
#     srcx, srcz = zip(*src_list)
#     # rec = time_loop(wave, c, nt, shots, srcx, srcz, dt, h, recz)
#     for it in tqdm.trange(nt):
#         u_now = addsources(u_now, shots, srcz, srcx, wave[it])
#         u_next = 2 * u_now - u_pre + (c * dt) ** 2 * laplace(u_now, h)
#         u_pre, u_now = u_now, u_next
#         rec = record(rec, it, u_now, shots, recz)
#     return rec
