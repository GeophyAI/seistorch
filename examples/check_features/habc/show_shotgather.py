import numpy as np
import matplotlib.pyplot as plt

"""Load the shot gather"""
cog_habc = np.load('shot_gather_habc.npy', allow_pickle=True)[3]
cog_pml = np.load('shot_gather_pml.npy', allow_pickle=True)[3]


"""Show the shot gather"""
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
vmin, vmax = np.percentile(cog_habc, [1, 99])
ax[0].imshow(cog_habc, vmin=vmin, vmax=vmax, cmap="seismic", aspect='auto')
ax[1].imshow(cog_pml, vmin=vmin, vmax=vmax, cmap="seismic", aspect='auto')
ax[0].set_title('HABC')
ax[1].set_title('PML')
plt.show()
