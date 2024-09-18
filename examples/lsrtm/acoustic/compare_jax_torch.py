import torch
import numpy as np
import matplotlib.pyplot as plt

true_m = np.load('./velocity/true_m.npy')
bwidth = 50
expand = 50
invt_torch = torch.load('./results/se_torch/model_F00E29.pt', 'cpu')['m']
invt_torch = invt_torch.detach().numpy()[bwidth:-bwidth, bwidth+expand:-bwidth-expand]

invt_jax = np.load('./results/se_jax/model_F00E30.npy')[1][bwidth:-bwidth, bwidth+expand:-bwidth-expand]
true_m = true_m[:, expand:-expand]

fig, axes= plt.subplots(1, 2, figsize=(10, 5))
vmin,vmax=np.percentile(true_m, [2, 98])
axes[0].imshow(invt_torch, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[0].set_title('Torch')
axes[1].imshow(invt_jax, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[1].set_title('Jax')
plt.tight_layout()
plt.savefig('figures/selsrtm.png', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(true_m[:, 200], 'c', label='True')
plt.plot(invt_torch[:, 200], 'r', label='Torch')
plt.plot(invt_jax[:, 200], 'b', label='Jax')
plt.legend()
plt.show()

