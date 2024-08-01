import numpy as np
import matplotlib.pyplot as plt
import torch

obs = np.load('results/l2_outlier/obs.npy')
syn = np.load('results/l2_outlier/syn.npy')[0]

fig, axes = plt.subplots(1, 2, figsize=(6, 4))
vmin, vmax = np.percentile(obs, [2, 98])
kwargs = {"cmap": "seismic", "aspect": "auto", "vmin": vmin, "vmax": vmax}
axes[0].imshow(obs, **kwargs)
axes[0].set_title("Observed")

axes[1].imshow(syn, **kwargs)
axes[1].set_title("Synthetic")