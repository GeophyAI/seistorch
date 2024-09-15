import numpy as np
import matplotlib.pyplot as plt
import torch

pmln=50
jax_grad = np.load("results/fwi_classic_ADPML_jax/gradient000.npy")[0, pmln:-pmln, pmln:-pmln]
torch_grad = torch.load("results/fwi_classic_ADPML/grad_vp_0.pt", 'cpu')[pmln:-pmln, pmln:-pmln]

fig, axes= plt.subplots(1, 2, figsize=(8, 4))
for d, ax, title in zip([jax_grad, torch_grad], axes.ravel(), ['Jax', 'Torch']):
    vmin, vmax=np.percentile(d, [5, 95])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.show()
