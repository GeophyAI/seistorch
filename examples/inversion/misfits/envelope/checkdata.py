import numpy as np
import matplotlib.pyplot as plt

obs = np.load('results/l2/obs0.npy')
syn = np.load('results/l2/syn0.npy')

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].imshow(obs[5], cmap='gray', aspect='auto')
axes[1].imshow(syn[5], cmap='gray', aspect='auto')
plt.tight_layout()
plt.show()