import sys, torch
sys.path.append('../../../../')
from seistorch.array import SeisArray
from seistorch.signal import filter
import matplotlib.pyplot as plt
import numpy as np

def read_h5(path, no):
    import h5py
    with h5py.File(path, 'r') as f:
        return f[f'shot_{no}'][:]

obs = read_h5('observed.hdf5', 0)

freqs = [1]
forder = 3
dt = 0.002
fobs_jax = SeisArray(obs).filter(dt, freqs, forder, axis=0)
fobs_torch = filter(torch.from_numpy(obs), dt, freqs, forder, btype='lowpass', axis=0)

fig,axes=plt.subplots(1,3,figsize=(9,6))
vmin,vmax=np.percentile(obs,[2, 98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(obs,**kwargs)
axes[0].set_title("Original")
vmin,vmax=np.percentile(fobs_jax,[2, 98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[1].imshow(fobs_jax,**kwargs)
axes[1].set_title("Jax")
vmin,vmax=np.percentile(fobs_torch,[2, 98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[2].imshow(fobs_torch,**kwargs)
axes[2].set_title("Torch")
plt.tight_layout()
plt.show()