import numpy as np
import matplotlib.pyplot as plt
from configures import dh, nz, nx
import os
from scipy.ndimage import gaussian_filter

# Load velocity
vp = np.ones((nz, nx), dtype=np.float32)*1500
vp[vp.shape[0]//2:, :] = 2000
smvp = vp.copy()
smvp[vp.shape[0]//2-5:vp.shape[0]//2+5, :] = 1750
# smvp[vp.shape[0]//2-10:vp.shape[0]//2, :] = 1750

# smvp = gaussian_filter(smvp, sigma=5)
# True reflectivity
true_m = 2*(vp-smvp)/smvp

# grids
nz, nx = true_m.shape

# Show
extent = [0, nx*dh, nz*dh, 0]
kwargs = dict(cmap="seismic", extent=extent, aspect="auto")
fig, axes=plt.subplots(1, 3, figsize=(12, 3))
axes[0].imshow(vp, vmin=1500, vmax=2000, **kwargs)
axes[0].set_title("True velocity")
axes[1].imshow(smvp, vmin=1500, vmax=2000, **kwargs)
axes[1].set_title("Initial velocity")
kwargs.update(cmap='gray')
axes[2].imshow(true_m, vmin=-0.5, vmax=0.5, **kwargs)
axes[2].set_title("True reflectivity")
for ax in axes:
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
plt.tight_layout()
plt.savefig("true_model.png")
plt.show()

os.makedirs("models", exist_ok=True)
np.save("models/true_vp.npy", vp)
np.save("models/true_reflectivity.npy", true_m)
np.save("models/smooth_vp.npy", smvp)



