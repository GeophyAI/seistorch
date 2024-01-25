import numpy as np
import matplotlib.pyplot as plt

pmln = 50

grad_ADPML = np.load("./results/fwi_classic_ADPML/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]
grad_ADHABC = np.load("./results/fwi_classic_ADHABC/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]

grad_BSPML = np.load("./results/fwi_classic_BSPML/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]
grad_HABC = np.load("./results/fwi_classic_BSHABC/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]

fig, axes= plt.subplots(2, 2, figsize=(8, 8))
for d, ax, title in zip([grad_ADPML, grad_BSPML, grad_ADHABC, grad_HABC], 
                        axes.ravel(), 
                        ['AD-PML', 'BS-PML', 'AD-HABC', 'BS-HABC']):
    vmin, vmax=np.percentile(d, [5, 95])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("compare_AD_BS.png")
plt.show()