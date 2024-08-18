import torch
import numpy as np
from configure import *
from utils import *
import matplotlib.pyplot as plt

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Load wavelet 
wavelet = ricker(np.arange(nt) * dt-delay*dt, f=fm)
fig,ax=plt.subplots(1,1,figsize=(5,4))
ax.plot(np.arange(nt)*dt, wavelet)
ax.set_title('Ricker Wavelet')
ax.set_xlabel('Time (s)')
plt.show()
show_freq_spectrum(wavelet.reshape(nt,1,1), dt=dt, end_freq=25, title='Frequency Spectrum')

# Load velocity
vp = np.load("models/vp.npy")
vs = np.load("models/vs.npy")
rho = np.load("models/rho.npy")

# Pad the velocity model
vp = np.pad(vp, ((npml, npml), (npml, npml)), mode='edge')
vs = np.pad(vs, ((npml, npml), (npml, npml)), mode='edge')
rho = np.pad(rho, ((npml, npml), (npml, npml)), mode='edge')

nz,nx = vp.shape
domain = vp.shape
pmlc = generate_pml_coefficients_2d(vp.shape, npml)

# Transfer to tensor
vp = to_tensor(vp, dev)
vs = to_tensor(vs, dev)
rho = to_tensor(rho, dev)
wavelet = wavelet.to(dev)
pmlc = pmlc.to(dev)

# Geometry
src_x = np.arange(npml, nx-npml, src_x_step)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]
kwargs = dict(wave=wavelet, parameters=[vp, vs, rho], pmlc=pmlc, src_list=sources, domain=domain, dt=dt, h=dh, dev=dev, recz=recz, npml=npml)
# Run forward simulation
with torch.no_grad():
    rec = forward(**kwargs)

show_gathers(rec.cpu().numpy()[:,0,:,:,], size=3, figsize=(8, 5))
show_gathers(rec.cpu().numpy()[:,1,:,:,], size=3, figsize=(8, 5))

np.save("obs.npy", rec.cpu().numpy())
print('Saved observed data to obs.npy')
