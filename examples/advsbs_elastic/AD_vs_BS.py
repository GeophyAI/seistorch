import numpy as np
import matplotlib.pyplot as plt

pmln = 50

grad_AD_vp = np.load("./results/fwi_classic_AD/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]
grad_AD_vs = np.load("./results/fwi_classic_AD/gradvsF00E00.npy")[pmln:-pmln, pmln:-pmln]

grad_BS_vp = np.load("./results/fwi_classic_BS/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]
grad_BS_vs = np.load("./results/fwi_classic_BS/gradvsF00E00.npy")[pmln:-pmln, pmln:-pmln]

fig, axes= plt.subplots(2, 2, figsize=(8, 8))
for d, ax, title in zip([grad_AD_vp, grad_AD_vs, 
                         grad_BS_vp, grad_BS_vs], axes.ravel(), 
                         ['AD-vp', 'AD-vs', 'BS-vp', 'BS-vs']):
    vmin, vmax=np.percentile(d, [2, 98])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("compare_AD_BS.png")
plt.show()