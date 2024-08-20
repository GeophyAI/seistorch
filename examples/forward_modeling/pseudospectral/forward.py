import torch, os, tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils_torch import forward, ricker, showgeom, show_gathers, generate_pml_coefficients_2d
from configure import *

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

l2loss = torch.nn.MSELoss()

domain = (nz+2*npml, nx+2*npml)

# load model
vel = np.load("models/vp.npy")
vel = np.pad(vel, ((npml, npml), (npml, npml)), mode='edge')

# pml coefficients
pmlc = generate_pml_coefficients_2d(vel.shape, npml)

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
src_x = np.linspace(npml, nx+npml, 20)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(npml, nx+npml, 1)
receiver_locz = np.ones_like(receiver_locx)*recz

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

# show geometry
showgeom(vel, sources, receivers, figsize=(5, 4))
print(f"The number of sources: {len(sources)}")
print(f"The number of receivers: {len(receivers[0])}")

# forward
#  for observed data
# To GPU
vel = torch.from_numpy(vel).float().to(dev)
pmlc = pmlc.to(dev)
kwargs = dict(wave=wave, src_list = np.array(sources), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, b=pmlc)
with torch.no_grad():
    print('Forward modeling using Pseudo-Spectral method')
    rec_obs_ps = forward(c=vel, ps=True, **kwargs)

    # Show gathers
    show_gathers(rec_obs_ps.cpu().numpy(), size=3, figsize=(5, 3), savepath='shotgather_ps.png')
    np.save("obs_ps.npy", rec_obs_ps.cpu().numpy())

    print('Forward modeling using Finite-Difference method')
    rec_obs_fd = forward(c=vel, ps=False, **kwargs)

    # Show gathers
    show_gathers(rec_obs_fd.cpu().numpy(), size=3, figsize=(5, 3), savepath='shotgather_fd.png')
    np.save("obs_fd.npy", rec_obs_fd.cpu().numpy())

#  for synthetic data
fig, ax=plt.subplots(1,1,figsize=(5,3))
ax.plot(rec_obs_ps[0].cpu().numpy()[:,0], 'r', label='PS')
ax.plot(rec_obs_fd[0].cpu().numpy()[:,0], 'b', label='FD')
ax.legend()
ax.set_title("Synthetic data")
plt.tight_layout()
plt.savefig("synthetic_data.png")
plt.show()

