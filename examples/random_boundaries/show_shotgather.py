import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader

"""
Configures
"""
config_path = "./forward.yml"
noboundary = "./noboundary.npy"
pmlboundary = "./pml.npy"
randomboundary = "./random.npy"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data
noboundary = np.load(noboundary, allow_pickle=True)[0]
pmlboundary = np.load(pmlboundary, allow_pickle=True)[0]
randomboundary = np.load(randomboundary, allow_pickle=True)[0]

nshots = 3
nsamples, ntraces, ncomponent = noboundary.shape
show_data = [noboundary, pmlboundary, randomboundary]
print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")
titles = ["No boundary", "PML boundary", "Random boundary"]
# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=nshots, figsize=(6, 3))
for i in range(nshots):
    vmin, vmax = np.percentile(show_data[i], [1, 99])
    kwargs = {"cmap": "seismic", 
              "aspect": "auto", 
              "vmin": vmin, 
              "vmax": vmax, 
              "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
    axes[i].imshow(show_data[i], **kwargs)
    axes[i].set_xlabel("x (m)")
    axes[i].set_ylabel("t (s)")
    axes[i].set_title(f"{titles[i]}")
plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()
