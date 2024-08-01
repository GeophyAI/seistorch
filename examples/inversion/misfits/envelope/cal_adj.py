import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from scipy.signal import hilbert
sys.path.append('../../../../')
from seistorch.loss import L2, Envelope
from seistorch.transform import envelope

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

obs = np.load('observed.npy', allow_pickle=True)
syn = np.load('initial.npy', allow_pickle=True)
nshots = obs.shape[0]
nsamples, ntraces, nchannels = syn[0].shape
print(f"nshots: {nshots}, nsamples: {nsamples}, ntraces: {ntraces}")

# show the observed and synthetic data
shot_no = 3
observed = obs[shot_no]
synthetic = syn[shot_no]
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

# Calculate the envelope of obs and syn
obs_envelope = envelope(to_tensor(observed)).numpy()
syn_envelope = envelope(to_tensor(synthetic)).numpy()
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
vmin, vmax = np.percentile(obs_envelope, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')
axes[0].imshow(obs_envelope, **kwargs)
axes[0].set_title('Envelope of Observed data')
axes[1].imshow(syn_envelope, **kwargs)
axes[1].set_title('Envelope of Initial data')
plt.tight_layout()
plt.savefig('figures/Envelopes_Profile.png', dpi=300, bbox_inches='tight')
plt.show()

# Trace show
trace_no = 25
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot(observed[:, trace_no], label='obs')
axes[0].plot(obs_envelope[:, trace_no], label='env of obs')
axes[1].plot(synthetic[:, trace_no], label='syn')
axes[1].plot(syn_envelope[:, trace_no], label='env of syn')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig('figures/Envelopes_Trace.png', dpi=300, bbox_inches='tight')
plt.show()

# calculate the envelope loss and l2 loss
envelope_diff = Envelope(method='subtract') # eq.5 in the paper
envelope_square = Envelope(method='square') # eq.6 in the paper
envelope_log = Envelope(method='log') # eq.7 in the paper (does not work)

l2_criterion = L2()

observed = torch.from_numpy(observed).float().unsqueeze(0)
synthetic = torch.from_numpy(synthetic).float().unsqueeze(0)
synthetic.requires_grad = True

loss_envelope_diff = envelope_diff(synthetic, observed)
loss_envelope_square = envelope_square(synthetic, observed)
loss_l2 = l2_criterion(synthetic, observed)

adj_envelope_diff = torch.autograd.grad(loss_envelope_diff, synthetic, create_graph=True)[0]
adj_envelope_square = torch.autograd.grad(loss_envelope_square, synthetic, create_graph=True)[0]
adj_l2 = torch.autograd.grad(loss_l2, synthetic, create_graph=True)[0]

adj_envelope_diff = adj_envelope_diff.detach().numpy().squeeze()
adj_envelope_square = adj_envelope_square.detach().numpy().squeeze()
adj_l2 = adj_l2.detach().numpy().squeeze()
# show the adjoint fields
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
vmin, vmax = np.percentile(adj_l2, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')
axes[0].imshow(adj_l2, **kwargs)
axes[0].set_title('Adj by L2')
axes[1].imshow(adj_envelope_diff, **kwargs)
axes[1].set_title('Env loss(difference)')
axes[2].imshow(adj_envelope_square, **kwargs)
axes[2].set_title('Env loss(square)')
plt.tight_layout()
plt.savefig('figures/Adjoint_sources.png', dpi=300, bbox_inches='tight')
plt.show()

# Frequency spectrum
kwargs = dict(dt=0.001, end_Freq=50)
freqs, amp_l2 = freq_spectrum(adj_l2, **kwargs)
_, amp_diff = freq_spectrum(adj_envelope_diff, **kwargs)
_, amp_square = freq_spectrum(adj_envelope_square, **kwargs)
fig, ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(freqs, amp_l2/amp_l2.max(), 'b', label='L2')
ax.plot(freqs, amp_diff/amp_diff.max(), 'r', label='Env diff')
ax.plot(freqs, amp_square/amp_square.max(), 'g', label='Env square')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Normalized Amplitude')
ax.legend()
plt.tight_layout()
plt.savefig('figures/adj_freq_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()

### Calculate the adjoint source by hand v.s. by AD

# factor1 = (syn_envelope-obs_envelope)/syn_envelope
# # factor2 = 2*(syn_envelope**2-obs_envelope**2)
# fs2 = factor1*(syn[shot_no]-hilbert(syn[shot_no], axis=0).imag)

# fig, axes = plt.subplots(1, 2, figsize=(6, 3))
# vmin, vmax = np.percentile(adj_envelope, [1, 99])
# axes[0].imshow(adj_envelope, vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')
# axes[0].set_title('Adj cal by AD')
# vmin, vmax = np.percentile(fs2, [1, 99])
# axes[1].imshow(fs2, vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')
# axes[1].set_title('Adj cal by hand')
# plt.tight_layout()
# plt.show()


