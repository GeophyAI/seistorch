import numpy as np
import matplotlib.pyplot as plt

syn = np.load("results/jax/syn049.npy")
obs = np.load("results/jax/obs049.npy")

shot_no = 6

fig,axes= plt.subplots(1, 2, figsize=(10, 5))
vmin, vmax= np.percentile(syn[shot_no], [2, 98])
axes[0].imshow(syn[shot_no], cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
axes[0].set_title("Synthetic")
axes[1].imshow(obs[shot_no], cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
axes[1].set_title("Observed")
plt.tight_layout()
plt.show()

# wavelet = np.load("results/jax/wavelet.npy")
# plt.plot(wavelet)
# plt.show()