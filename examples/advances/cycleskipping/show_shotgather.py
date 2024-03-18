import numpy as np
import matplotlib.pyplot as plt
import sys, tqdm
sys.path.append("../../")
from yaml import load
from yaml import CLoader as Loader
np.random.seed(20230915)

from seistorch.signal import filter
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

nshots = obs.shape[0]
nsamples, ntraces, ncomponent = obs[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")


# Generate data mask
dmask = np.empty_like(obs)
mask = np.zeros_like(obs[0])

arrival_trace_first = 0.256 # s
arrival_trace_last = 2.0 # s
arrivals = np.linspace(arrival_trace_first, arrival_trace_last, ntraces)//cfg['geom']['dt']
arrivals = arrivals.astype(int)

for trace in range(ntraces):
    avl = arrivals[trace]
    upper = avl-200 if avl-200 > 0 else 0
    mask[upper:avl+120, trace, :] = 1

for shot in range(nshots):
    dmask[shot] = mask
np.save("datamask.npy", dmask)
# show 5 shots randomly
showshots = np.random.randint(0, nshots, 5)
# Plot the data
rec_idx = np.arange(ntraces)
fig, axes = plt.subplots(nrows=1, ncols=showshots.size, figsize=(12, 6))
for ax, shot_no in zip(axes.ravel(), showshots.tolist()):
    vmin, vmax = np.percentile(obs[shot_no], [2, 98])
    kwargs = {"cmap": "seismic", 
            "aspect": "auto", 
            "vmin": vmin, 
            "vmax": vmax, 
            "extent": [0, ntraces*cfg['geom']['h'], nsamples*cfg['geom']['dt'], 0]}
    d = filter(np.expand_dims(obs[shot_no], 0), cfg['geom']['dt'], N=3, freqs=(5,15), axis=0, mode="bandpass")
    d = d[0]*dmask[shot_no]
    ax.imshow(d[..., 0], **kwargs)
    ax.scatter(rec_idx*cfg['geom']['h'], arrivals*cfg['geom']['dt'], s=1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title(f"Shot {shot_no}")
plt.tight_layout()
plt.savefig("shot_gather.png", dpi=300)
plt.show()



