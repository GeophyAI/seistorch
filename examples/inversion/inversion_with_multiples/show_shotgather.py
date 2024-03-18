import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader
np.random.seed(20230915)

import sys
sys.path.append("../..")
from seistorch.io import SeisIO
io = SeisIO(load_cfg=False)
"""
Configures
"""
config_path = "./forward_nomultiple.yml"
obs_nomultiple_Path = "./observed_nomultiple.npy"
obs_withmultiple_Path = "./observed_withmultiple.npy"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data
obs_nomultiple = np.load(obs_nomultiple_Path, allow_pickle=True)
obs_withmultiple = np.load(obs_withmultiple_Path, allow_pickle=True)

# nshots = 86
nshots = obs_nomultiple.shape[0]
nsamples, ntraces, ncomponent = obs_nomultiple[0].shape

print(f"The data has {0} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")
showshots = [obs_nomultiple[40], obs_withmultiple[40]]
# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
for ax, d in zip(axes.ravel(), showshots):
    vmin, vmax = np.percentile(d, [2, 98])
    kwargs = {"cmap": "seismic", 
            "aspect": "auto", 
            "vmin": vmin, 
            "vmax": vmax, 
            "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
    ax.imshow(d[..., 0], **kwargs)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
axes[0].set_title("Observed data without multiples")
axes[1].set_title("Observed data with multiples")
plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()

# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
