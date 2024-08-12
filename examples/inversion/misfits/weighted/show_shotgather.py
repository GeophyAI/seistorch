import numpy as np
import matplotlib.pyplot as plt
import h5py
from yaml import load
from yaml import CLoader as Loader
np.random.seed(20230915)
"""
Configures
"""
config_path = "./configs/forward.yml"
obsPath = "./shot_gather.hdf5"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data

# show 5 shots randomly
showshots = np.random.randint(0, 10, 5)
# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=showshots.size, figsize=(12, 6))
for ax, shot_no in zip(axes.ravel(), showshots.tolist()):
    with h5py.File(obsPath, 'r') as f:
        obs = f[f'shot_{shot_no}'][:]
    nsamples, ntraces, _ = obs.shape
    vmin, vmax = np.percentile(obs, [2, 98])
    kwargs = {"cmap": "seismic", 
            "aspect": "auto", 
            "vmin": vmin, 
            "vmax": vmax, 
            "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
    ax.imshow(obs[..., 0], **kwargs)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title(f"Shot {shot_no}")
plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()