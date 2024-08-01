import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from configure import *

torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Load velocity
vel = np.load("../../models/marmousi_model/true_vp.npy")
vel = vel[:, expand:-expand][::model_scale, ::model_scale]
vel = np.pad(vel, ((pmln, pmln), (pmln, pmln)), mode="edge")
pmlc = generate_pml_coefficients_2d(vel.shape, N=pmln, multiple=False)
domain = vel.shape
nz, nx = domain
# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
# show spectrum
plt.figure(figsize=(5, 3))
amp = np.abs(np.fft.fft(wave.cpu().numpy()))
freqs = np.fft.fftfreq(nt, dt)
amp = amp[freqs >= 0]
freqs = freqs[freqs >= 0]
plt.plot(freqs[:50], amp[:50])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Geometry
src_x = np.arange(pmln, nx-pmln, srcx_step)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(pmln, nx-pmln, 1)
receiver_locz = np.ones_like(receiver_locx)*recz

# The receivers are fixed
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

# show geometry
showgeom(vel, sources, receivers, figsize=(5, 4))
print(f"The number of sources: {len(sources)}")
print(f"The number of receivers: {len(receivers[0][0])}")

# forward
#  for observed data
# To GPU
kwargs = dict(wave=wave, b=pmlc, src_list = np.array(sources), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, pmln=pmln)
with torch.no_grad():
    vel = torch.from_numpy(vel).float().to(dev)
    rec_obs = forward(c=vel, **kwargs)
# Show gathers
show_gathers(rec_obs.cpu().numpy(), figsize=(10, 6))
# Save on disk
np.save("obs.npy", rec_obs.cpu().numpy())

print("Observed data saved as obs.npy")

