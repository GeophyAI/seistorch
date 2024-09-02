import torch
from functools import partial
import sys
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from configures import *

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load velocity
smvp = np.load("models/smooth_vp.npy")
true_m = np.load("models/true_reflectivity.npy")

# Padding
smvp = np.pad(smvp, ((bwidth, bwidth), (bwidth, bwidth)), mode="edge")
true_m = np.pad(true_m, ((bwidth, bwidth), (bwidth, bwidth)), mode="edge")

# Get the shape of the model
domain = smvp.shape
nz, nx = domain

# HABC coefficients
pmlc = generate_pml_coefficients_2d(domain, N=bwidth).to(dev)

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
srcxs = np.arange(bwidth, nx-bwidth, srcx_step).tolist()
srczs = (np.ones_like(srcxs) * srcz).tolist()
src_loc = list(zip(srcxs, srczs))

recxs = np.arange(bwidth, nx-bwidth, 1).tolist()
reczs = (np.ones_like(recxs) * recz).tolist()
rec_loc = list(zip(recxs, reczs))

# show geometry
showgeom(smvp, src_loc, rec_loc, figsize=(5, 4))
print(f"The number of sources: {len(src_loc)}")
print(f"The number of receivers: {len(rec_loc)}")

# forward for observed data
# To GPU
smvp = torch.from_numpy(smvp).float().to(dev)
true_m = torch.from_numpy(true_m).float().to(dev)

kwargs = dict(b=pmlc, src_list=np.array(src_loc), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, bwidth=bwidth)
# Run forward
with torch.no_grad():
    rec_born = forward(wave, true_m, smvp, **kwargs)
# Show gathers``
show_gathers(rec_born.cpu().numpy(), figsize=(10, 6))

np.save("born.npy", rec_born.cpu().numpy())