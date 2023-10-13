import numpy as np
import matplotlib.pyplot as plt

import sys, tqdm
sys.path.append("../..")
from seistorch.show import SeisShow
from seistorch.io import SeisIO
from seistorch.signal import filter

show = SeisShow()
io = SeisIO(load_cfg=False)
cfg = io.read_cfg("./config/forward_obs.yml")
freqs = cfg["geom"]["multiscale"][0]

obs = np.load("./observed.npy", allow_pickle=True)
ini = np.load("./observed_init.npy", allow_pickle=True)

obs = filter(obs, dt=cfg['geom']['dt'], N=3, freqs=freqs, axis=0)
ini = filter(ini, dt=cfg['geom']['dt'], N=3, freqs=freqs, axis=0)

nshots = obs.shape[0]
nsamples, ntraces, ncomponent = obs[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

dt=0.001
# Generate data mask
dmask = np.empty_like(obs)
mask = np.zeros_like(obs[0])

arrival_trace_first = 0.2 # s
arrival_trace_last = 3.8 # s
arrivals = np.linspace(arrival_trace_first, arrival_trace_last, ntraces)//dt
arrivals = arrivals.astype(int)

for trace in range(ntraces):
    avl = arrivals[trace]
    upper = avl-800 if avl-800 > 0 else 0
    mask[upper:avl+350, trace, :] = 1

for shot in range(nshots):
    dmask[shot] = mask

shot_no = 50
# plt.imshow(dmask[0].squeeze(), aspect="auto")
# plt.show()

show.shotgather([obs[shot_no].squeeze(), 
                 obs[shot_no].squeeze()*dmask[0].squeeze(), 
                 ini[shot_no].squeeze()*dmask[0].squeeze()], 
                ["observed", "Masked obs", "Masked Init"], 
                aspect="auto",
                inarow=True)

np.save("datamask.npy", dmask)