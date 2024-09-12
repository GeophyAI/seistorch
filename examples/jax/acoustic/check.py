import numpy as np
import matplotlib.pyplot as plt
import torch

npml = 50
invt_jax = np.load("results/jax/inverted199.npy")[0, npml:-npml, npml:-npml]
invt_torch = torch.load("results/torch/model_199.pt")['vp'].cpu().numpy()
invt_torch = invt_torch[npml:-npml, npml:-npml]
fig, axes= plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(invt_jax, cmap="seismic", vmin=1500, vmax=5500, aspect="auto")
axes[0].set_title("JAX")
axes[1].imshow(invt_torch, cmap="seismic", vmin=1500, vmax=5500, aspect="auto")
axes[1].set_title("Torch")
plt.show()


####################
grad_jax = np.load("results/jax/gradient000.npy")[0, npml:-npml, npml:-npml]
grad_torch = torch.load("results/torch/grad_vp_0.pt").cpu().numpy()[npml:-npml, npml:-npml]

fig,axes= plt.subplots(1, 2, figsize=(10, 5))
vmin, vmax= np.percentile(grad_jax, [2, 98])
plt.colorbar(axes[0].imshow(grad_jax, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto"), ax=axes[0])
axes[0].set_title("JAX")
vmin, vmax= np.percentile(grad_torch, [2, 98])
plt.colorbar(axes[1].imshow(grad_torch, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto"), ax=axes[1])
axes[1].set_title("Torch")
plt.tight_layout()
plt.show()