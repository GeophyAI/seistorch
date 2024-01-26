import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader
np.random.seed(20230915)
import h5py

import sys
sys.path.append("/home/shaowinw/seistorch")
from seistorch.io import SeisIO
io = SeisIO(load_cfg=False)
"""
Configures
"""
btype = 'habc'
config_path = f"./config/{btype}.yml"
obsPath = f"./observed_{btype}.hdf5"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data

# nshots = 86


# show 5 shots randomly
showshots = np.random.randint(0, 86, 5)
# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=showshots.size, figsize=(12, 6))
for ax, shot_no in zip(axes.ravel(), showshots.tolist()):
    with h5py.File(cfg['geom']['obsPath']) as F:
        d = F[f'shot_{shot_no}'][:]
    nsamples, ntraces, ncomponent = d.shape
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
plt.savefig(f"shot_gather_{btype}.png", dpi=300)
plt.show()