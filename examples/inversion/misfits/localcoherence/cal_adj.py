import numpy as np
import matplotlib.pyplot as plt
import torch
import sys, os, h5py
sys.path.append('../../../../')
from seistorch.loss import L2, LocalCoherence

os.makedirs('figures', exist_ok=True)

def to_tensor(d):
    return torch.from_numpy(d).float()
def to_np(d):
    return d.detach().numpy()
def freq_spectrum(d, dt, end_Freq=25):
    freqs = np.fft.fftfreq(d.shape[0], dt)
    amp = np.sum(np.abs(np.fft.fft(d, axis=0)), axis=(1))
    freqs = freqs[:len(freqs)//2]
    amp = amp[:len(amp)//2]
    amp = amp[freqs<end_Freq]
    freqs = freqs[freqs<end_Freq]
    return freqs, amp
def read_hdf5(path, shot_no):
    with h5py.File(path, 'r') as f:
        data = f[f'shot_{shot_no}'][:]
    return data


observed = read_hdf5('observed.hdf5', 3)
synthetic = read_hdf5('initial.hdf5', 3)
nsamples, ntraces, nchannels = observed.shape

# show the observed and synthetic data
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
vmin, vmax = np.percentile(observed, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')
axes[0].imshow(observed, **kwargs)
axes[0].set_title('Observed data')
axes[1].imshow(synthetic, **kwargs)
axes[1].set_title('Initial data')
plt.tight_layout()
plt.savefig('figures/Profiles.png', dpi=300, bbox_inches='tight')
plt.show()

observed = torch.from_numpy(observed).float().unsqueeze(0)
synthetic = torch.from_numpy(synthetic).float().unsqueeze(0)
synthetic.requires_grad = True

l2_criterion = L2()
l2_loss = l2_criterion(observed,synthetic)
lc_criterion = LocalCoherence(wt=51, wx=31, sigma_hx=21.0, sigma_tau=11.0)
lc_loss = lc_criterion(observed, synthetic)

adj_l2 = torch.autograd.grad(l2_loss, synthetic, create_graph=True)[0]
adj_lc = torch.autograd.grad(lc_loss, synthetic, create_graph=True)[0]

adj_l2 = adj_l2.detach().numpy().squeeze()
adj_lc = adj_lc.detach().numpy().squeeze()
# show the adjoint fields
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
vmin, vmax = np.percentile(adj_l2, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')
axes[0].imshow(adj_l2, **kwargs)
axes[0].set_title('Adj by L2')
axes[1].imshow(adj_lc, **kwargs)
axes[1].set_title('Adj by Local coherence')
plt.tight_layout()
plt.savefig('figures/Adjoint_sources.png', dpi=300, bbox_inches='tight')
plt.show()
