import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader
np.random.seed(20230915)

import sys
sys.path.append("../../../..")
from seistorch.io import SeisIO
io = SeisIO(load_cfg=False)
"""
Configures
"""
config_path = "./forward.yml"
obsPath = "./observed.npy"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data
obs = np.load(obsPath, allow_pickle=True)

# nshots = 86
nshots = obs.shape[0]
nsamples, ntraces, ncomponent = obs[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

# show 5 shots randomly
showshots = np.random.randint(0, nshots, 5)
# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=showshots.size, figsize=(12, 6))
for ax, shot_no in zip(axes.ravel(), showshots.tolist()):
    d = obs[shot_no]
    vmin, vmax = np.percentile(d, [2, 98])
    kwargs = {"cmap": "seismic", 
            "aspect": "auto", 
            "vmin": vmin, 
            "vmax": vmax, 
            "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
    ax.imshow(d[..., 0], **kwargs)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title(f"Shot {shot_no}")
plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()