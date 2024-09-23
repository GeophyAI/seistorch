import numpy as np
import matplotlib.pyplot as plt
import torch

npml = 50
expand = 50
true_vp = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
true_vs = np.load('../../../models/marmousi_model/true_vs.npy')[:,expand:-expand]

init_vp = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
init_vs = np.load('../../../models/marmousi_model/linear_vs.npy')[:,expand:-expand]

# Torch
inverted = torch.load('./results/torch/model_F01E49.pt')
inverted_vp = inverted['vp'].detach().cpu().numpy()[npml:-npml, npml+expand:-npml-expand]
inverted_vs = inverted['vs'].detach().cpu().numpy()[npml:-npml, npml+expand:-npml-expand]

# Jax
# inverted = np.load('./results/jax/model_F03E49.npy')
# inverted_vp = inverted[0][npml:-npml, npml+expand:-npml-expand]
# inverted_vs = inverted[1][npml:-npml, npml+expand:-npml-expand]

nz, nx = true_vp.shape
vmin_vp, vmax_vp = true_vp.min(), true_vp.max()
vmin_vs, vmax_vs = true_vs.min(), true_vs.max()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))
titles = ["True vp", "True vs", "Init vp", "Init vs", "Inverted vp", "Inverted vs"]
for d, ax, title in zip([true_vp, true_vs,
                        init_vp, init_vs, 
                        inverted_vp, inverted_vs], axes.ravel(), titles):
    if "vp" in title:
        vmin, vmax = vmin_vp, vmax_vp
    else:
        vmin, vmax = vmin_vs, vmax_vs
    kwargs = dict(cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax, extent=[0, nx, nz, 0])
    _ax_ = ax.imshow(d, **kwargs)
    plt.colorbar(_ax_, ax=ax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("Inverted.png", dpi=300)
plt.show()

gradient = np.load('./results/jax/gradient_F00E10.npy')
gradient_vp = gradient[0][npml:-npml, npml+expand:-npml-expand]
gradient_vs = gradient[1][npml:-npml, npml+expand:-npml-expand]
fig, axes= plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
titles = ["Gradient vp", "Gradient vs"]
for d, ax, title in zip([gradient_vp, gradient_vs], axes.ravel(), titles):
    if "vp" in title:
        vmin, vmax = np.percentile(gradient_vp, [5, 95])
    else:
        vmin, vmax = np.percentile(gradient_vs, [5, 95])
    kwargs = dict(cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax, extent=[0, nx, nz, 0])
    _ax_ = ax.imshow(d, **kwargs)
    plt.colorbar(_ax_, ax=ax)
    ax.set_title(title)
plt.tight_layout()
plt.show()

syn = np.load('./results/jax/syn_49.npy')[0]
obs = np.load('./results/jax/obs_49.npy')

fig, axes = plt.subplots(1,2, figsize=(5, 3))
vmin,vmax=np.percentile(syn, [5, 95])
kwargs = dict(cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax, extent=[0, nx, nz, 0])
axes[0].imshow(syn[...,0], **kwargs)
axes[0].set_title("Synthetic")
axes[1].imshow(obs[...,0], **kwargs)
axes[1].set_title("Observed")
plt.tight_layout()
plt.show()
