import torch
from functools import partial

torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

def regularization_model(m, x=True, z=True):
    l=0.
    if z:
        l += torch.sum(0.5*(m[1:] - m[:-1])**2)
    if x:
        l += torch.sum(0.5*(m[:,1:] - m[:,:-1])**2)
    return l

import numpy as np
import matplotlib.pyplot as plt
from utils import *

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configure
model_scale = 2 # 1/2
expand = 50
expand = int(expand/model_scale)
delay = 150 # ms
fm = 10 # Hz
dt = 0.001 # s
nt = 1500 # timesteps
dh = 10 # m
pmln = 50
srcz = 5+pmln # grid point
recz = 5+pmln # grid point

# Training
criterion = torch.nn.MSELoss()
lr = 20.
epochs = 51

# Load velocity
vel = np.load("../../models/marmousi_model/true_vp.npy")
init = np.load("../../models/marmousi_model/linear_vp.npy")
vel = vel[::model_scale,::model_scale]
init = init[::model_scale,::model_scale]
vel = np.pad(vel, ((pmln, pmln), (pmln, pmln)), mode="edge")
init = np.pad(init, ((pmln, pmln), (pmln, pmln)), mode="edge")
pmlc = generate_pml_coefficients_2d(vel.shape, N=pmln, multiple=False)


domain = vel.shape
nz, nx = domain
# imshow(vel, vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 4))

# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
tt = np.arange(nt) * dt
plt.plot(tt, wave.cpu().numpy())
plt.title("Wavelet")
plt.show()
# Frequency spectrum
# Show freq < 10Hz
freqs = np.fft.fftfreq(nt, dt)[:nt//2]
amp = np.abs(np.fft.fft(wave.cpu().numpy()))[:nt//2]
amp = amp[freqs <= 30] 
freqs = freqs[freqs <= 30]
plt.plot(freqs, amp)
plt.title("Frequency spectrum")
plt.show()
# Geometry
srcxs = np.arange(expand+pmln, nx-expand-pmln, 1).tolist()
srczs = (np.ones_like(srcxs) * srcz).tolist()
src_loc = list(zip(srcxs, srczs))

recxs = np.arange(expand+pmln, nx-expand-pmln, 1).tolist()
reczs = (np.ones_like(recxs) * recz).tolist()
rec_loc = list(zip(recxs, reczs))

# show geometry
showgeom(vel, src_loc, rec_loc, figsize=(5, 4))
print(f"The number of sources: {len(src_loc)}")
print(f"The number of receivers: {len(rec_loc)}")

# forward for observed data
# To GPU
vel = torch.from_numpy(vel).float().to(dev)
with torch.no_grad():
    rec_obs = forward(wave, vel, pmlc, np.array(src_loc), domain, dt, dh, dev, recz, pmln)
# Show gathers
show_gathers(rec_obs.cpu().numpy(), figsize=(10, 6))


# forward for initial data
# To GPU
init = torch.from_numpy(init).float().to(dev)
init.requires_grad = True
# Configures for training
opt = torch.optim.Adam([init], lr=lr)

def closure(vp, use_reg=True):
    opt.zero_grad()
    rand_size = 8
    rand_shots = np.random.randint(0, len(src_loc), size=rand_size).tolist()
    rec_syn = forward(wave, vp, pmlc, np.array(src_loc)[rand_shots], domain, dt, dh, dev, recz, pmln)
    loss = criterion(rec_syn, rec_obs[rand_shots])
    if use_reg:
        loss += 1e-9 * regularization_model(vp, x=True, z=False)
    loss.backward()
    return loss

Loss = []
for epoch in tqdm.trange(epochs):
    loss = opt.step(partial(closure, vp=init, use_reg=True))
    Loss.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        imshow(init.cpu().detach().numpy()[pmln:-pmln,pmln:-pmln], vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 3))
        plt.show()

    # if epoch == 0:
    #     grad = init.grad.cpu().numpy()[pmln:-pmln,pmln:-pmln]
    #     grad[:12] = 0
    #     vmin, vmax=np.percentile(grad, [1, 99])
    #     imshow(grad, vmin=vmin, vmax=vmax, cmap="seismic", figsize=(5, 3))
    #     plt.show()
    # break

plt.plot(Loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()