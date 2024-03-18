import numpy as np
import matplotlib.pyplot as plt

obs = np.load("./results/obs.npy")
syn = np.load("./results/syn.npy")
adj = np.load("./results/adj.npy")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

vmin, vmax=np.percentile(obs,[2,98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(obs.squeeze(), **kwargs)
axes[0].set_title("Observed data")

vmin, vmax=np.percentile(syn,[2,98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[1].imshow(syn.squeeze(), **kwargs)
axes[1].set_title("Synthetic data")

vmin, vmax=np.percentile(adj,[2,98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[2].imshow(adj.squeeze(), **kwargs)
axes[2].set_title("Adjoint data")
plt.tight_layout()
plt.show()
