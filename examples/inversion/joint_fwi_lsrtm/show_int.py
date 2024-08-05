import numpy as np
import matplotlib.pyplot as plt

obs = np.load('results/obs.npy')
syn = np.load('results/syn.npy')[0]

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
vmin, vmax = np.percentile(obs, [2, 98])
axes[0].imshow(obs, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
axes[0].set_title('Observed')
axes[1].imshow(syn, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
axes[1].set_title('Synthetic')
plt.tight_layout()
plt.show()


