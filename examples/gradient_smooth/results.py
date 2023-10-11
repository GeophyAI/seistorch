import numpy as np
import matplotlib.pyplot as plt

pmln = 50

grad_sm = np.load("./results/with_grad_sm/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]
grad_nosm = np.load("./results/no_grad_sm/gradvpF00E00.npy")[pmln:-pmln, pmln:-pmln]

fig, axes= plt.subplots(1, 2, figsize=(8, 4))
for d, ax, title in zip([grad_nosm, grad_sm], axes.ravel(), ['Gradient', 'Smoothed gradient']):
    vmin, vmax=np.percentile(d, [5, 95])
    ax.imshow(d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("compare.png")
plt.show()