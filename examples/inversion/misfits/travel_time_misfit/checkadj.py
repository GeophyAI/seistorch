import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/shaowinw/seistorch")
from seistorch.show import SeisShow
from seistorch.signal import travel_time_diff
from seistorch.loss import Loss
from seistorch.io import SeisIO
from seistorch.signal import SeisSignal

io = SeisIO(load_cfg=False)
show = SeisShow()

obs = np.load("./observed.npy", allow_pickle=True)
syn = np.load("./observed_init.npy", allow_pickle=True)
mask = np.load("./datamask.npy", allow_pickle=True)

cfg = io.read_cfg("./config/forward_obs.yml")
ss = SeisSignal(cfg)
freqs = cfg["geom"]["multiscale"][0]
obs = ss.filter(obs, freqs=freqs, axis=0)
syn = ss.filter(syn, freqs=freqs, axis=0)


syn_nomask = np.stack(syn[0:1], axis=0)
obs_nomask = np.stack(obs[0:1], axis=0)
obs = np.stack(obs[0:1], axis=0)*np.stack(mask[0:1], axis=0)
syn = np.stack(syn[0:1], axis=0)*np.stack(mask[0:1], axis=0)
mak = np.stack(mask[0:1], axis=0)
criterion = Loss("traveltime").loss(None)

obs = torch.from_numpy(obs).cuda()
syn = torch.from_numpy(syn).cuda()
syn.requires_grad = True

loss = criterion(syn, obs)

adj = torch.autograd.grad(loss, syn, create_graph=True)[0]

shot_no = 0

show.shotgather([obs[shot_no].cpu().detach().numpy(), 
                 syn[shot_no].cpu().detach().numpy(),
                 adj[shot_no].cpu().detach().numpy()],
                ["Observed", "Synthetic", "Adjoint"],
                inarow=True,
                dt=0.001,
                normalize=False,
                savepath="./checkadj.png",
                dx=20)

_syn = syn[shot_no].cpu().detach().numpy()

adj_byhand = _syn.copy()
adj_byhand[1:] = (_syn[1:] - _syn[:-1])

for trace in range(obs.shape[2]):
    tt = travel_time_diff(obs[shot_no][:,trace, 0 ], 
                               syn[shot_no][:,trace, 0]).cpu().numpy()
    adj_byhand[:, trace, :] *= tt
# fig,ax=plt.subplots(1,1)
# plt.plot(tt)
# plt.show()
 
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
adj_auto = adj[shot_no].cpu().detach().numpy()
vmin,vmax=np.percentile(adj_auto, [5, 95])
axes[0].imshow(adj_auto, vmin=vmin, vmax=vmax, aspect="auto", cmap="seismic")
axes[0].set_title("Auto-grad")
vmin,vmax=np.percentile(adj_byhand, [5, 95])
axes[1].imshow(adj_byhand, vmin=vmin, vmax=vmax, aspect="auto", cmap="seismic")
axes[1].set_title("By-hand")
plt.show()
