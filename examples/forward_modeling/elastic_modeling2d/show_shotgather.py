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
fig, axes = plt.subplots(nrows=2, ncols=nshots, figsize=(8, 4))
for i in range(nshots):
    for c in range(ncomponent):
        vmin, vmax = np.percentile(obs[i][..., c], [1, 99])
        kwargs = {"cmap": "seismic", 
                    "aspect": "auto", 
                    "vmin": vmin, 
                    "vmax": vmax, 
                    "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
        axes[c, i].imshow(obs[i][..., c], **kwargs)
        axes[c, i].set_xlabel("x (m)")
        axes[c, i].set_ylabel("t (s)")
        component = "vx"if c==0 else "vz"
        axes[c, i].set_title(f"Shot {i+1} {component}")
plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()
