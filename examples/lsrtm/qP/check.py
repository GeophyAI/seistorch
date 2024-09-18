import numpy as np
import matplotlib.pyplot as plt

obs = np.load('results/vti_bornobs_jax/obs_10.npy')
syn = np.load('results/vti_bornobs_jax/syn_10.npy')[0]

fig,axes = plt.subplots(1,2,figsize=(10,5))
vmin,vmax=np.percentile(obs, [2, 98])
axes[0].imshow(obs, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[0].set_title('Observed')
vmin,vmax=np.percentile(syn, [2, 98])
axes[1].imshow(syn, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[1].set_title('Synthetic')
plt.tight_layout()
plt.show()
