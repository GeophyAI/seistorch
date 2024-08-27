import numpy as np
import matplotlib.pyplot as plt

obs = np.load('results_traditional_cg/obs0.npy')
syn = np.load('results_traditional_cg/syn0.npy')
no = 5
fig, axes = plt.subplots(1,2,figsize=(6,3))
vmin, vmax=np.percentile(obs[no], [2, 98])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[0].imshow(obs[no], **kwargs)
axes[0].set_title('Observed')
axes[1].imshow(syn[no], **kwargs)
axes[1].set_title('Synthetic')
plt.tight_layout()
plt.show()

