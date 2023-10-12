import numpy as np
import matplotlib.pyplot as plt

import sys, tqdm
sys.path.append("../..")
from seistorch.show import SeisShow

show = SeisShow()

obs = np.load("./observed.npy", allow_pickle=True)

nshots = obs.shape[0]
nsamples, ntraces, ncomponent = obs[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

dt=0.001
# Generate data mask
dmask = np.empty_like(obs)
mask = np.zeros_like(obs[0])

arrival_trace_first = 0.5 # s
arrival_trace_last = 2.05 # s
arrivals = np.linspace(arrival_trace_first, arrival_trace_last, ntraces)//dt
arrivals = arrivals.astype(int)

for trace in range(ntraces):
    avl = arrivals[trace]
    upper = avl if avl > 0 else 0
    mask[upper:, trace, :] = 1

for shot in range(nshots):
    dmask[shot] = mask

shot_no = 70

show.shotgather([obs[shot_no].squeeze(),
                 obs[shot_no].squeeze()*dmask[0].squeeze()],
                ["observed", "Masked obs"],
                aspect="auto",
                inarow=True)

np.save("datamask.npy", dmask)