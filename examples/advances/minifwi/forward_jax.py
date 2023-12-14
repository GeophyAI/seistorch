import jax.numpy as jnp
from jax import grad, jit, vmap
import time

import matplotlib.pyplot as plt
from utils_jax import forward, ricker
from utils_torch import imshow, showgeom, show_gathers
from jax.profiler import trace

# configure
model_scale = 2 # 1/2
expand = 50
expand = int(expand/model_scale)
delay = 150 # ms
fm = 3 # Hz
dt = 0.001 # s
nt = 2000 # timesteps
dh = 20.0 # m
srcz = 0 # grid point
recz = 0 # grid point
# Training
lr = 10.
epochs = 100

# Load velocity
vel = jnp.load("../../models/marmousi_model/true_vp.npy")
init = jnp.load("../../models/marmousi_model/linear_vp.npy")
vel = vel[::model_scale,::model_scale]
init = init[::model_scale,::model_scale]
domain = vel.shape
nz, nx = domain
# imshow(vel, vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 4))

# load wave
wave = ricker(jnp.arange(nt) * dt-delay*dt, f=fm)

# Geometry
srcx = jnp.arange(expand, nx-expand, 1)
srcz = (jnp.ones_like(srcx) * srcz)
src_loc = list(zip(srcx.tolist(), srcz.tolist()))

recx = jnp.arange(expand, nx-expand, 1)
recz = (jnp.ones_like(recx) * recz)
rec_loc = list(zip(recx.tolist(), recz.tolist()))

# show geometry
showgeom(vel, src_loc, rec_loc, figsize=(5, 4))
print(f"The number of sources: {len(src_loc)}")
print(f"The number of receivers: {len(rec_loc)}")

rec_obs = forward(wave, vel, jnp.array(src_loc), domain, dt, dh, 0)
show_gathers(rec_obs, figsize=(8, 5))
# Dump the trace data to a file
# Print the trace data
# forward for observed data
# To GPU
# vel = torch.from_numpy(vel).float().to(dev)
# with torch.no_grad():
#     rec_obs = forward(wave, vel, np.array(src_loc), domain, dt, dh, dev, recz=0)
# # Show gathers
# show_gathers(rec_obs.cpu().numpy(), figsize=(10, 10))


# # forward for initial data
# # To GPU
# init = torch.from_numpy(init).float().to(dev)
# init.requires_grad = True
# # Configures for training
# opt = torch.optim.Adam([init], lr=lr)

# def closure():
#     opt.zero_grad()
#     rand_size = 8
#     rand_shots = np.random.randint(0, len(src_loc), size=rand_size).tolist()
#     rec_syn = forward(wave, init, np.array(src_loc)[rand_shots], domain, dt, dh, dev, recz=0)
#     loss = criterion(rec_syn, rec_obs[rand_shots])
#     loss.backward()
#     return loss

# for epoch in range(epochs):
#     loss = opt.step(closure)
#     print(f"Epoch: {epoch}, Loss: {loss}")
#     if epoch % 10 == 0:
#         imshow(init.cpu().detach().numpy(), vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 3))