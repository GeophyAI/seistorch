import sys, torch
sys.path.append("../../../..")
import numpy as np
import matplotlib.pyplot as plt
from seistorch.loss import L2, Wasserstein1d, Wasserstein_test
dev = "cuda" if torch.cuda.is_available() else "cpu"
# Load the data from disk
obs = np.load("./obs.npy", allow_pickle=True)
syn = np.load("./syn.npy", allow_pickle=True)
print(f'The data has {obs.shape[0]} shots,n{obs[0].shape[0]} time samples, {obs[0].shape[1]} traces, and {obs[0].shape[2]} components.')
# select a shot
shot = 5
obs = np.expand_dims(obs[shot], axis=0)
syn = np.expand_dims(syn[shot], axis=0)
# convert the data to torch tensor
obs = torch.from_numpy(obs).to(dev)
syn = torch.from_numpy(syn).to(dev)
syn.requires_grad = True
# different loss functions
l2loss = L2()
# otloss = Wasserstein1d()
otloss = Wasserstein_test()

# Calculate the adjoint source of l2 loss function
adj_l2 = torch.autograd.grad(l2loss(syn, obs), 
                             syn, 
                             create_graph=True, 
                             retain_graph=False)[0].detach().cpu().numpy()[0]
# Calculate the adjoint source of OT loss function
adj_ot = torch.autograd.grad(otloss(syn, obs), 
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
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
vmin, vmax = np.percentile(adj_l2, [1, 99])
axes[0].imshow(adj_l2, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[0].set_title("L2 Adjoint")
vmin, vmax = np.percentile(adj_ot, [1, 99])
axes[1].imshow(adj_ot, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[1].set_title("OT Adjoint")
plt.show()

fig, ax = plt.subplots()
ax.plot(_obs[:,100], label="Observed")
ax.plot(_syn[:,100], label="Synthetic")
ax.plot(adj_l2[:,100], label="L2 Adjoint")
ax.plot(adj_ot[:,100], label="OT Adjoint")
ax.legend()
plt.show()



