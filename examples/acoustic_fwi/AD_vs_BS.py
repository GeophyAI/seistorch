import numpy as np
import matplotlib.pyplot as plt

pmln = 50

grad_AD = np.load("./results/fwi_classic_AD/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]
grad_BS = np.load("./results/fwi_classic_BS/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]

fig, axes= plt.subplots(1, 2, figsize=(8, 4))
for d, ax, title in zip([grad_AD, grad_BS], axes.ravel(), ['AD', 'BS']):
    vmin, vmax=np.percentile(d, [5, 95])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("compare_AD_BS.png")
plt.show()