import numpy as np
import matplotlib.pyplot as plt

syn = np.load('./results_traditional_adam/syn0.npy')[0]
obs = np.load('./results_traditional_adam/obs0.npy')[0]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
vmin, vmax = np.percentile(obs, [2, 98])
kwargs = {"cmap": "seismic", 
        "aspect": "auto", 
        "vmin": vmin, 
        "vmax": vmax}
# add colorbar
cbar = plt.colorbar(axes[0].imshow(obs, **kwargs), ax=axes[0])
vmin, vmax = np.percentile(syn, [2, 98])
kwargs = {"cmap": "seismic", 
        "aspect": "auto", 
        "vmin": vmin, 
        "vmax": vmax}
cbar = plt.colorbar(axes[1].imshow(syn, **kwargs), ax=axes[1])
axes[0].set_title("Observation")
axes[1].set_title("Synthetic")
plt.tight_layout()
plt.show()