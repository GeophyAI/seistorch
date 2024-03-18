import torch
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import imshow, showgeom, show_gathers, Propagator, ricker
import timeit

# torch.cuda.cudnn_enabled = False
# torch.backends.cudnn.benchmark = True
# configure
model_scale = 2 # 1/2
expand = 50
expand = int(expand/model_scale)
delay = 150 # ms
fm = 3 # Hz
dt = 0.001 # s
nt = 2000 # timesteps
dh = 20 # m
srcz = 0 # grid point
recz = 0 # grid point
# Training
# criterion = torch.nn.MSELoss()
lr = 10.
epochs = 100

# Load velocity
vel = np.load("../../models/marmousi_model/true_vp.npy")
init = np.load("../../models/marmousi_model/linear_vp.npy")
vel = vel[::model_scale,::model_scale]
init = init[::model_scale,::model_scale]
domain = vel.shape
nz, nx = domain
# imshow(vel, vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 4))

# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)

# Geometry
srcx = np.arange(expand, nx-expand, 10).tolist()
srcz = (np.ones_like(srcx) * srcz).tolist()
src_loc = list(zip(srcx, srcz))

recx = np.arange(expand, nx-expand, 1).tolist()
recz = (np.ones_like(recx) * recz).tolist()
rec_loc = list(zip(recx, recz))

# show geometry
showgeom(vel, src_loc, rec_loc, figsize=(5, 4))
print(f"The number of sources: {len(src_loc)}")
print(f"The number of receivers: {len(rec_loc)}")


wave_torch = Propagator(backend="torch")
wave_jax= Propagator(backend="jax")

# with torch
# with torch.no_grad():
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     vel_torch = torch.from_numpy(vel).float().to(dev).unsqueeze(0)
#     kwargs = dict(wave=wave, c=vel_torch, src_list=np.array(src_loc), domain=domain, dt=dt, h=dh, dev=dev, recz=0)

#     rec_obs = wave_torch.forward(**kwargs)
#     show_gathers(rec_obs.cpu().numpy(), figsize=(8, 5))
#     # tt = timeit.timeit(lambda: wave_torch.forward(**kwargs), number=10)

# with jax
dev = None
vel_jax = jnp.array(vel)
vel_jax = jnp.expand_dims(vel_jax, axis=0)
kwargs = dict(wave=wave, c=vel_jax, src_list=src_loc, domain=domain, dt=dt, h=dh, dev=dev, recz=0)
rec_obs = wave_jax.forward(**kwargs)
show_gathers(rec_obs, figsize=(8, 5))