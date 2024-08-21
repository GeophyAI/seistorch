import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys, tqdm
sys.path.append("../../..")
from seistorch.show import SeisShow

def read_hdf5(path, shot_no=0):
    with h5py.File(path, 'r') as f:
        return f[f"shot_{shot_no}"][:]

show = SeisShow()

obs = read_hdf5("observed.hdf5", 70)

nshots = 87
nsamples, ntraces, ncomponent = obs.shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

dt=0.001
# Generate data mask
dmask = np.zeros((nshots, nsamples, ntraces, ncomponent))
mask = np.zeros_like(obs)

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


show.shotgather([obs.squeeze(),
                 obs.squeeze()*dmask[0].squeeze()],
                ["observed", "Masked obs"],
                inarow=True)

# write to hdf5
with h5py.File(f"mask.hdf5", 'w') as f:
    pass
for i in tqdm.tqdm(range(nshots)):
    with h5py.File(f"mask.hdf5", 'a') as f:
        f.create_dataset(f"shot_{i}", data=dmask[i])