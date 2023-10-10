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

obs = np.load("./observed.npy", allow_pickle=True)
syn = np.load("./observed_init.npy", allow_pickle=True)
mask = np.load("./datamask.npy", allow_pickle=True)

obs = np.stack(obs[0:1], axis=0)*np.stack(mask[0:1], axis=0)
syn = np.stack(syn[0:1], axis=0)*np.stack(mask[0:1], axis=0)

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
                dx=20)


# nb, nt, nr, nc = obs.shape
# tt = torch.zeros((nb, nr, nc), dtype=torch.float32)
# tt_args = torch.zeros((nb, nr, nc), dtype=torch.float32)
# padding = nt - 1
# indices = torch.arange(2*nt-1)
# scale = 1e6
# dt = 0.001

# def softargmax1d(input, beta=100):
#     *_, n = input.shape
#     input = F.softmax(beta * input, dim=-1)
#     indices = torch.linspace(0, 1, n).to(input.device)
#     result = torch.sum((n - 1) * input * indices, dim=-1)
#     return result

# for b in range(nb):
#     for r in range(nr):
#         for c in range(nc):
#             _x = obs[b,:,r,c]
#             _y = syn[b,:,r,c]

#             if torch.max(torch.abs(_x)) >0 and torch.max(torch.abs(_y)) >0:

#                 cc = F.conv1d(_x.unsqueeze(0), _y.unsqueeze(0).unsqueeze(0), padding=padding)

#                 # max_indices = torch.argmax(cc, dim=1)

#                 # one_hot = torch.zeros_like(cc)
#                 # one_hot.scatter_(1, max_indices.view(-1, 1), 1.0)
#                 # max_index = (cc * one_hot).sum()
                
#                 # loss += (max_index-nt+1)*dt

#                 # Method 2
#                 # logits = F.gumbel_softmax(cc*scale, tau=1, hard=True)
#                 # max_index = torch.sum(indices.to(logits.device) * logits)

#                 # Method3
#                 max_index = softargmax1d(cc, beta=1)
#                 tt[b,r,c] = (max_index-nt+1)*dt

#                 # tt_args[b,r,c] = travel_time_diff(_x, _y)#(torch.argmax(cc, dim=1)-nt+1)*dt

# plt.plot(tt[shot_no].cpu().detach().numpy().squeeze())
# # plt.plot(tt_args[shot_no].cpu().detach().numpy().squeeze())
# plt.show()