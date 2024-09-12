import numpy as np
import matplotlib.pyplot as plt
import torch

npml = 50
invt = np.load("results/jax/inverted049.npy")[0, npml:-npml, npml:-npml]
vmin, vmax= np.percentile(invt, [2, 98])
plt.figure(figsize=(5, 3))
plt.imshow(invt, cmap="seismic", aspect="auto")
plt.show()

# Show Gradients
grad_jax = np.load("results/jax/gradient000.npy")
grad_jax_vp = grad_jax[0, npml:-npml, npml:-npml]
grad_jax_vs = grad_jax[1, npml:-npml, npml:-npml]

grad_torch_vp = torch.load("results/torch/grad_vp_0.pt").cpu().numpy()[npml:-npml, npml:-npml]
grad_torch_vs = torch.load("results/torch/grad_vs_0.pt").cpu().numpy()[npml:-npml, npml:-npml]

fig,axes= plt.subplots(2, 2, figsize=(8, 8))
vmin, vmax= np.percentile(grad_jax_vp, [2, 98])
axes[0, 0].imshow(grad_jax_vp, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
axes[0, 0].set_title("JAX VP")
vmin, vmax= np.percentile(grad_torch_vp, [2, 98])
axes[0, 1].imshow(grad_torch_vp, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
axes[0, 1].set_title("Torch VP")
vmin, vmax= np.percentile(grad_jax_vs, [2, 98])
axes[1, 0].imshow(grad_jax_vs, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
axes[1, 0].set_title("JAX VS")
vmin, vmax= np.percentile(grad_torch_vs, [2, 98])
axes[1, 1].imshow(grad_torch_vs, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
axes[1, 1].set_title("Torch VS")
plt.tight_layout()
plt.show()

# import torch
# npml = 50
# inverted = torch.load("results/torch/model_49.pt")['vp'].cpu().numpy()
# inverted = inverted[npml:-npml, npml:-npml]
# plt.figure(figsize=(5, 3))
# plt.imshow(inverted, cmap="seismic", vmin=1500, vmax=5500, aspect="auto")
# plt.show()

# grad_jax = np.load("results/jax/gradient000.npy")[0, npml:-npml, npml:-npml]
# grad_torch = torch.load("results/torch/grad_vp_0.pt").cpu().numpy()[npml:-npml, npml:-npml]

# fig,axes= plt.subplots(1, 2, figsize=(10, 5))
# vmin, vmax= np.percentile(grad_jax, [2, 98])
# plt.colorbar(axes[0].imshow(grad_jax, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto"), ax=axes[0])
# axes[0].set_title("JAX")
# vmin, vmax= np.percentile(grad_torch, [2, 98])
# plt.colorbar(axes[1].imshow(grad_torch, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto"), ax=axes[1])
# axes[1].set_title("Torch")
# plt.tight_layout()
# plt.show()