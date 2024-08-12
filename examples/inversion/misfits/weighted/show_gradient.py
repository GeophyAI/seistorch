import numpy as np
import matplotlib.pyplot as plt
import torch

bwitdh = 50

grad_weighted = torch.load("./results/weighted/grad_vp_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]
grad_l2 = torch.load("./results/l2/grad_vp_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]
grad_envelope = torch.load("./results/envelope/grad_vp_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]

fig, axes= plt.subplots(1, 3, figsize=(8, 4))
for d, ax, title in zip([grad_l2, grad_envelope, grad_weighted], 
                        axes.ravel(), 
                        ['L2', 'Envelope', 'Weighted']):
    vmin, vmax=np.percentile(d, [5, 95])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("compare_AD_BS.png")
plt.show()