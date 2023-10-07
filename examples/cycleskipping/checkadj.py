import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.append("../../")
from seistorch.show import SeisShow
from seistorch.signal import travel_time_diff
from seistorch.loss import Loss

show = SeisShow()

obs = np.load("./results/towed_withoutlow_ttcc/obs.npy")
syn = np.load("./results/towed_withoutlow_ttcc/syn.npy")
# adj = np.load("./results/towed_withoutlow_ttcc/adj.npy")

criterion = Loss("ttcc").loss(None)

obs = torch.from_numpy(obs).cuda()
syn = torch.from_numpy(syn).cuda()
syn.requires_grad = True

loss = criterion(syn, obs)

adj = torch.autograd.grad(loss, syn, create_graph=True)[0]

shot_no = 0

show.shotgather([obs[shot_no].cpu().detach().numpy()[:,64:128], 
                 syn[shot_no].cpu().detach().numpy()[:,64:128], 
                 adj[shot_no].cpu().detach().numpy()[:,64:128]],
                ["Observed", "Synthetic", "Adjoint"],
                inarow=True,
                dt=0.001,
                aspect="auto",
                dx=20)
# adj = adj.cpu().detach().numpy()
# vmin,vmax=np.percentile(adj[1], [0,100])
# plt.imshow(adj[1], vmin=vmin, vmax=vmax, aspect="auto")
# plt.show()

# nb, nt, nr, nc = obs.shape
# tt = torch.zeros((nb, nr, nc), dtype=torch.float32)
# padding = nt - 1
# indices = torch.arange(2*nt-1)
# scale = 1e6
# dt = 0.001
# for b in range(nb):
#     for r in range(nr):
#         for c in range(nc):
#             _x = obs[b,:,r,c]
#             _y = syn[b,:,r,c]

#             if torch.max(torch.abs(_x)) >0 and torch.max(torch.abs(_y)) >0:

#                 cc = F.conv1d(_x.unsqueeze(0), _y.unsqueeze(0).unsqueeze(0), padding=padding)
#                 logits = F.gumbel_softmax(cc*scale, tau=1, hard=True)
#                 max_index = torch.sum(indices * logits)
#                 tt[b,r,c] = (max_index-nt+1)*dt

# plt.plot(tt[shot_no].squeeze())
# plt.show()

# print(tt[shot_no])
# show.wiggle([obs[shot_no], syn[shot_no], adj[shot_no]],
#             ["red", "black", "blue"],
#             ["Observed", "Synthetic", "Adjoint"],
#             dt=0.001,
#             dx=20,
#             downsample=2)


