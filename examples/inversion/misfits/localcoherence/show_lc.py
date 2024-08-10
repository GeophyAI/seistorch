import torch, sys, h5py
import matplotlib.pyplot as plt
import numpy as np
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sys.path.append('../../../../')
from seistorch.signal import local_coherence

def ricker(fm=10,nt=1000,dt=0.002):
    t = torch.arange(0, nt*dt, dt)-0.5
    t0 = 1/fm
    r = (1-2*(torch.pi**2)*(fm**2)*(t-t0)**2)*torch.exp(-(torch.pi**2)*(fm**2)*(t-t0)**2)
    return r

def read_hdf5(path, shot_no):
    with h5py.File(path, 'r') as f:
        data = f[f'shot_{shot_no}'][:]
    return data

# data shape : (batch, nsamples, ntraces, nchannels)
x = ricker().view(1, 1000, 1, 1).to(dev)
x = x.repeat(1, 1, 128, 1)
y = x.clone()

lc = local_coherence(x, y, wt=11, wx=11, sigma_hx=21.0, sigma_tau=11.0)
lc = lc.squeeze().cpu().numpy()

fig, axes=plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(x.squeeze().cpu().numpy(), aspect='auto', cmap='jet')
axes[0].set_title('X')
axes[1].imshow(y.squeeze().cpu().numpy(), aspect='auto', cmap='jet')
axes[1].set_title('Y')
axes[2].imshow(lc, aspect='auto', cmap='jet', vmin=-1, vmax=1)
axes[2].set_title('Local Coherence')
plt.tight_layout()
plt.savefig('figures/LocalCoherence_Ricker.png', dpi=300, bbox_inches='tight')
plt.show()

obs = read_hdf5('observed.hdf5', 3)
syn = read_hdf5('initial.hdf5', 3)
obs = torch.from_numpy(obs).float().to(dev).unsqueeze(0)
syn = torch.from_numpy(syn).float().to(dev).unsqueeze(0)

lc = local_coherence(obs, syn, wt=201, wx=11, sigma_hx=1.0, sigma_tau=1.0)
lc = lc.squeeze().cpu().numpy()
fig, axes=plt.subplots(1, 3, figsize=(10, 3))
obs = obs.squeeze().cpu().numpy()
vmin,vmax=np.percentile(obs,[1,99])
kwargs=dict(vmin=vmin,vmax=vmax,cmap='gray',aspect='auto')
axes[0].imshow(obs, **kwargs)
axes[0].set_title('Observed')
axes[1].imshow(syn.squeeze().cpu().numpy(), **kwargs)
axes[1].set_title('Synthetic')
axes[2].imshow(lc, aspect='auto', cmap='jet', vmin=-1, vmax=1)
axes[2].set_title('Local Coherence')
plt.tight_layout()
plt.savefig('figures/LocalCoherence_Shots.png', dpi=300, bbox_inches='tight')
plt.show()
