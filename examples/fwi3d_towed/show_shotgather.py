import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader

"""
Configures
"""
config_path = "./forward.yml"
obsPath = "./shot_gather.npy"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data
obs = np.load(obsPath, allow_pickle=True)

nshots = obs.shape[0]
nsamples, ntraces, ncomponent = obs[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

showshots = np.array([i for i in range(9)])#np.random.randint(0, obs.shape[0], 8)

# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=showshots.size, figsize=(12, 4))
for idx, ax in enumerate(axes.ravel()):   
    vmin, vmax = np.percentile(obs[showshots[idx]], [2, 98])
    kwargs = {"cmap": "seismic", 
            "aspect": "auto", 
            "vmin": vmin, 
            "vmax": vmax, 
            "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
    ax.imshow(obs[showshots[idx]][..., 0], **kwargs)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title(f"Shot Gather {showshots[idx]}")

plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()