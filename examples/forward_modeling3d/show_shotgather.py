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

# Plot the data
fig, ax = plt.subplots(nrows=1, ncols=nshots, figsize=(12, 4))
    
vmin, vmax = np.percentile(obs[0], [2, 98])
kwargs = {"cmap": "seismic", 
          "aspect": "auto", 
          "vmin": vmin, 
          "vmax": vmax, 
          "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
ax.imshow(obs[0][..., 0], **kwargs)
ax.set_xlabel("x (m)")
ax.set_ylabel("t (s)")
ax.set_title(f"Shot Gather")

plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()