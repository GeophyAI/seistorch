import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.append("../../")
from seistorch.show import SeisShow
from seistorch.signal import travel_time_diff
from seistorch.loss import Loss
from seistorch.io import SeisIO
from seistorch.signal import filter

io = SeisIO(load_cfg=False)
show = SeisShow()

obs = np.load("./observed.npy", allow_pickle=True)
syn = np.load("./observed_init.npy", allow_pickle=True)
mask = np.load("./datamask.npy", allow_pickle=True)

cfg = io.read_cfg("./config/forward_obs.yml")
freqs = cfg["geom"]["multiscale"][0]
obs = filter(obs, dt=cfg['geom']['dt'], N=3, freqs=freqs, axis=0)
syn = filter(syn, dt=cfg['geom']['dt'], N=3, freqs=freqs, axis=0)


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
                aspect="auto",
                savepath="./checkadj.png",
                dx=20)