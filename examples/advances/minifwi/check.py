import numpy as np
import matplotlib.pyplot as plt

epoch = 1000
unit = 0.001
true = np.load("../../models/marmousi_model/true_vp.npy")*unit
true = true[::2,::2]
invt = np.load(f"results/vel_{epoch:04d}.npy")#*unit

fig,axes=plt.subplots(1, 2, figsize=(6, 3))
kwargs=dict(vmin=1500*unit, vmax=5500*unit, cmap="seismic", aspect="auto")
axes[0].imshow(true, **kwargs)
axes[1].imshow(invt, **kwargs)
fig.savefig(f"vel_P{epoch:04d}.png", dpi=300)

fig,ax=plt.subplots(1, 1, figsize=(5, 3))
ax.plot(true[:, 100], label="true")
ax.plot(invt[:, 100], label="inverted")
ax.legend()
fig.savefig(f"vel_{epoch:04d}.png", dpi=300)