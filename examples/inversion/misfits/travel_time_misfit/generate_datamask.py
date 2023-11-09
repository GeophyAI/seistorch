import numpy as np
import matplotlib.pyplot as plt

import sys, tqdm
sys.path.append("../../../..")
from seistorch.show import SeisShow
from seistorch.io import SeisIO
from seistorch.signal import SeisSignal

show = SeisShow()
io = SeisIO(load_cfg=False)
cfg = io.read_cfg("./config/forward_obs.yml")
freqs = cfg["geom"]["multiscale"][0]
ss = SeisSignal(cfg)

obs = np.load("./observed.npy", allow_pickle=True)
ini = np.load("./observed_init.npy", allow_pickle=True)

obs = ss.filter(obs, freqs=freqs, axis=0)
ini = ss.filter(ini, freqs=freqs, axis=0)

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
    mask[upper:avl+300, trace, :] = 1

for shot in range(nshots):
    dmask[shot] = mask

shot_no = 50
# plt.imshow(dmask[0].squeeze(), aspect="auto")
# plt.show()

show.wiggle([obs[shot_no], 
             obs[shot_no]*dmask[0], 
             ini[shot_no]*dmask[0]], 
             ["r", "b", "g"], 
             ["observed", "Masked obs", "Masked Init"], 
             dt=cfg['geom']['dt'],
             dx=cfg['geom']['h'], 
             downsample=20,
             savepath="masked_data.png")

np.save("datamask.npy", dmask)