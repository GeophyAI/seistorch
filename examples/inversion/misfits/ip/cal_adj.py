import sys, torch
sys.path.append("../../../..")
import numpy as np
import matplotlib.pyplot as plt
from seistorch.loss import L2, InstantaneousPhase, ExpInstantaneousPhase, InstantaneousPhase2
import h5py, os
savepath = r'figures'
os.makedirs(savepath, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
def read_hdf5(path, shot_no=0):
    with h5py.File(path, 'r') as f:
        data = f[f'shot_{shot_no}'][:]
    return data
# Load the data from disk
shot = 5
obs = read_hdf5("./obs.hdf5", shot)
syn = read_hdf5("./syn.hdf5", shot)
# select a shot
obs = np.expand_dims(obs, axis=0)
syn = np.expand_dims(syn, axis=0)
# convert the data to torch tensor
obs = torch.from_numpy(obs).to(dev)
syn = torch.from_numpy(syn).to(dev)
syn.requires_grad = True
# different loss functions
l2loss = L2()
iploss = InstantaneousPhase2()
# iploss = ExpInstantaneousPhase()
# iploss = InstantaneousPhase()

# Calculate the adjoint source of l2 loss function
adj_l2 = torch.autograd.grad(l2loss(syn, obs), 
                             syn, 
                             create_graph=True, 
                             retain_graph=False)[0].detach().cpu().numpy()[0]
# Calculate the adjoint source of IP loss function
adj_ip = torch.autograd.grad(iploss(syn, obs), 
                             syn, 
                             create_graph=True, 
                             retain_graph=False)[0].detach().cpu().numpy()[0]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
_obs = obs[0].detach().cpu().numpy()
vmin, vmax = np.percentile(_obs, [1, 99])
axes[0].imshow(_obs, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[0].set_title("Observed")
_syn = syn[0].detach().cpu().numpy()
vmin, vmax = np.percentile(_syn, [1, 99])
axes[1].imshow(_syn, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[1].set_title("Synthetic")
plt.tight_layout()
plt.savefig(f'{savepath}/obs_syn.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
vmin, vmax = np.percentile(adj_l2, [1, 99])
axes[0].imshow(adj_l2, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[0].set_title("L2 Adjoint")
vmin, vmax = np.percentile(adj_ip, [1, 99])
axes[1].imshow(adj_ip, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[1].set_title("IP Adjoint")
plt.tight_layout()
plt.savefig(f'{savepath}/adjoints.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(_obs[:,100], label="Observed")
ax.plot(_syn[:,100], label="Synthetic")
ax.plot(adj_l2[:,100], label="L2 Adjoint")
ax.plot(adj_ip[:,100], label="IP Adjoint")
ax.legend()
plt.tight_layout()
plt.savefig(f'{savepath}/adjoints_line.png', dpi=300, bbox_inches='tight')
plt.show()