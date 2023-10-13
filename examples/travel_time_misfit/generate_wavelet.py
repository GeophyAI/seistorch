import numpy as np
import torch
import sys
import os
sys.path.append("../..")
from seistorch.signal import ricker_wave
from seistorch.io import SeisIO
import matplotlib.pyplot as plt

cfg = SeisIO(load_cfg=False).read_cfg("./config/forward_obs.yml")

# Define the parameters
fm = cfg["geom"]["fm"]
dt = cfg["geom"]["dt"]
nt = cfg["geom"]["nt"]
wavelet_delay = cfg["geom"]["wavelet_delay"]

# Generate the wavelet
w1 = ricker_wave(fm, dt, nt, delay=wavelet_delay)
w2 = torch.diff(w1, prepend=torch.Tensor([0.]))

w1 = w1.numpy()
w2 = w2.numpy()

w2 = w2/np.max(np.abs(w2))

os.makedirs("./wavelet", exist_ok=True)
np.save("./wavelet/ricker.npy", w1)
np.save("./wavelet/ricker_diff.npy", w2)

fig, ax = plt.subplots(figsize=(8, 4))
t=np.arange(0, nt*dt, dt)
ax.plot(t, w1, label="w1")
ax.plot(t, w2, label="w2")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
plt.show()
fig.savefig("./wavelet/wavelet.png")

