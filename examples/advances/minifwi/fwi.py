import torch
import numpy as np
import matplotlib.pyplot as plt
import lesio
from utils import imshow, forward, ricker, showgeom, show_gathers

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
criterion = torch.nn.MSELoss()
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

# forward for observed data
# To GPU
vel = torch.from_numpy(vel).float().to(dev)
with torch.no_grad():
    rec_obs = forward(wave, vel, np.array(src_loc), domain, dt, dh, dev, recz=0)
# Show gathers
show_gathers(rec_obs.cpu().numpy(), figsize=(10, 10))


# forward for initial data
# To GPU
init = torch.from_numpy(init).float().to(dev)
init.requires_grad = True
# Configures for training
opt = torch.optim.Adam([init], lr=lr)

def closure():
    opt.zero_grad()
    rand_size = 8
    rand_shots = np.random.randint(0, len(src_loc), size=rand_size).tolist()
    rec_syn = forward(wave, init, np.array(src_loc)[rand_shots], domain, dt, dh, dev, recz=0)
    loss = criterion(rec_syn, rec_obs[rand_shots])
    loss.backward()
    return loss

for epoch in range(epochs):
    loss = opt.step(closure)
    print(f"Epoch: {epoch}, Loss: {loss}")
    if epoch % 10 == 0:
        imshow(init.cpu().detach().numpy(), vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 3))