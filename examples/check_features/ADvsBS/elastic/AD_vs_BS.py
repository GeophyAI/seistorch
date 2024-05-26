import numpy as np
import matplotlib.pyplot as plt
import torch

bwitdh = 50

grad_vp_ADPML = torch.load("./results/fwi_classic_AD/grad_vp_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]
grad_vs_ADPML = torch.load("./results/fwi_classic_AD/grad_vs_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]
grad_rho_ADPML = torch.load("./results/fwi_classic_AD/grad_rho_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]

grad_vp_BSPML = torch.load("./results/fwi_classic_BS/grad_vp_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]
grad_vs_BSPML = torch.load("./results/fwi_classic_BS/grad_vs_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]
grad_rho_BSPML = torch.load("./results/fwi_classic_BS/grad_rho_0.pt", 'cpu')[bwitdh:-bwitdh, bwitdh:-bwitdh]

fig, axes= plt.subplots(2, 3, figsize=(12, 8))
for d, ax, title in zip([grad_vp_ADPML, grad_vs_ADPML, grad_rho_ADPML, grad_vp_BSPML, grad_vs_BSPML, grad_rho_BSPML], 
                        axes.ravel(), 
                        ['AD-vp', 'AD-vs', 'AD-rho', 'BS-vp', 'BS-vs', 'BS-rho']):
    vmin, vmax=np.percentile(d, [5, 95])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("compare_AD_BS.png")
plt.show()