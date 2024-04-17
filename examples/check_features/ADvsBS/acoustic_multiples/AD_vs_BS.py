import numpy as np
import matplotlib.pyplot as plt
import torch

bwitdh = 50

grad_ADPML = torch.load("./results/fwi_classic_ADPML/grad_vp_0.pt", 'cpu')[:-bwitdh, bwitdh:-bwitdh]
grad_ADHABC = torch.load("./results/fwi_classic_ADHABC/grad_vp_0.pt", 'cpu')[:-bwitdh, bwitdh:-bwitdh]

grad_BSPML = torch.load("./results/fwi_classic_BSPML/grad_vp_0.pt", 'cpu')[:-bwitdh, bwitdh:-bwitdh]
grad_BSHABC = torch.load("./results/fwi_classic_BSHABC/grad_vp_0.pt", 'cpu')[:-bwitdh, bwitdh:-bwitdh]

fig, axes= plt.subplots(2, 2, figsize=(8, 8))
for d, ax, title in zip([grad_ADPML, grad_BSPML, grad_ADHABC, grad_BSHABC], 
                        axes.ravel(), 
                        ['AD-PML', 'BS-PML', 'AD-HABC', 'BS-HABC']):
    vmin, vmax=np.percentile(d, [5, 95])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("compare_AD_BS.png")
plt.show()
