import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, tqdm, time
import jax.numpy as jnp
from jax import grad
import numpy as np
import matplotlib.pyplot as plt
from utils_jax import *
from jax.example_libraries.optimizers import adam
import jax.random as random
rng_key = random.PRNGKey(20240905)
from functools import partial
from configure import *
os.makedirs("figures", exist_ok=True)

# Shardings
# Set up the device mesh
devices_count = len(jax.devices())
devices = jnp.arange(devices_count)
dist_sharding, replicated_sharding = get_sharding()
# Load velocity
vel = np.load(true_path)
init = np.load(init_path)
vel = vel[::model_scale, ::model_scale]
init = init[::model_scale, ::model_scale]
vel = np.pad(vel, ((pmln, pmln), (pmln, pmln)), mode="edge")
init = np.pad(init, ((pmln, pmln), (pmln, pmln)), mode="edge")
pmlc = generate_pml_coefficients_2d(vel.shape, N=pmln, multiple=False).numpy()
pmlc = jnp.array(pmlc)
# plt.imshow(pmlc, cmap="jet", aspect="auto")
# plt.colorbar()
# plt.show()
domain = vel.shape
nz, nx = domain

wave = ricker(jnp.arange(nt) * dt - delay * dt, f=fm)
tt = np.arange(nt) * dt
# plt.plot(tt, wave)
# plt.title("Wavelet")
# plt.show()

# Frequency spectrum
freqs = np.fft.fftfreq(nt, dt)[:nt//2]
amp = np.abs(np.fft.fft(wave))[:nt//2]
amp = amp[freqs <= 20] 
freqs = freqs[freqs <= 20]
# plt.plot(freqs, amp)
# plt.title("Frequency spectrum")
# plt.show()

# Geometry
srcxs = np.arange(expand+pmln, nx-expand-pmln-1, srcx_step).tolist()
srczs = (np.ones_like(srcxs) * srcz).tolist()
src_loc = list(zip(srcxs, srczs))

recxs = np.arange(expand+pmln, nx-expand-pmln, 1).tolist()
reczs = (np.ones_like(recxs) * recz).tolist()
rec_loc = list(zip(recxs, reczs))

# Show geometry
# showgeom(vel, src_loc, rec_loc, figsize=(5, 4))
print(f"The number of sources: {len(src_loc)}")
print(f"The number of receivers: {len(rec_loc)}")
"""###################################################"""
"""######################Forward######################"""
"""###################################################"""
kwargs = dict(b=pmlc, domain=domain, dt=dt, h=dh, recz=recz, pmln=pmln)
start_time = time.time()
rec_obs = forward(wave, vel, src_list=jnp.array(src_loc), **kwargs)
rec_obs.block_until_ready()
end_time = time.time()
print(f"Forward modeling time: {end_time - start_time:.2f}s")
# show_gathers(rec_obs, figsize=(10, 6))
"""#####################################################"""
"""######################Inversion######################"""
"""#####################################################"""

# Training Loop
init = jnp.array(init)

def loss(vp, shot_nums):
    shot_nums = jnp.array(shot_nums)
    syn = forward(wave, vp, src_list=jnp.array(src_loc)[shot_nums], **kwargs)
    return jnp.mean((syn - rec_obs[shot_nums])**2)

def compute_gradient(vp, shot_nums=[1, 2, 3]):
    return jax.value_and_grad(loss)(vp, jnp.array(shot_nums))

compute_gradient = jax.pmap(compute_gradient, axis_name='devices')

# @partial(jax.jit, static_argnames=['batch_size'])
# @partial(jax.pmap, axis_name='devices')
def fwi_step(vp, step, opt_state, rng_key=None, batch_size = 8):
    rng_key, subkey = random.split(rng_key)
    rand_shots = random.randint(subkey, (batch_size,), 0, len(src_loc))
    
    vp = jax.pmap(lambda _: vp)(devices)
    rand_shots = rand_shots.reshape(devices_count, -1)
    rand_shots = jax.device_put(rand_shots, dist_sharding)

    _loss, gradient = compute_gradient(vp, rand_shots)
    gradient = jnp.mean(gradient, axis=0)
    opt_state = opt_update(step, gradient, opt_state)
    return _loss, opt_state, rng_key

opt_init, opt_update, get_params = adam(lr)
opt_state = opt_init(init)
LOSS = []
for epoch in tqdm.trange(EPOCHS):
    _loss, opt_state, rng_key = fwi_step(get_params(opt_state), epoch, opt_state, rng_key, batch_size)
    LOSS.append(_loss)
    
    if epoch % show_every == 0:
        # show vel
        inverted = get_params(opt_state)[pmln:-pmln, pmln:-pmln]
        plt.figure(figsize=(5, 3))
        plt.imshow(inverted, vmin=1500., vmax=5500., cmap="seismic", aspect="auto")
        plt.colorbar()
        plt.savefig(f"figures/{epoch:03d}.png")
        plt.close()

        # show loss
        plt.plot(LOSS)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.close()

# # show loss
# plt.plot(LOSS)
# plt.title("Loss")
# plt.xlabel("Epoch")
# plt.show()