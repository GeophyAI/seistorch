import numpy as np
import matplotlib.pyplot as plt
import torch
from siren import Siren

from configure import *

domain = (nz+2*npml, nx+2*npml)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tempM = Siren(in_features=in_features, 
              out_features=out_features, 
              hidden_features=hidden_features, 
              hidden_layers=hidden_layers, 
              outermost_linear=True,
              domain_shape=domain).to(dev)

true_vp = np.load("models/vp.npy")
true_vs = np.load("models/vs.npy")

std_vp, mean_vp = 1000., 3000.
std_vs, mean_vs = std_vp/vp_vs_ratio, mean_vp/vp_vs_ratio

coords = tempM.coords.to(dev)
imvel = torch.load("results/model_3500.pt", dev)

vp = imvel(coords)[0][..., 0]
vs = imvel(coords)[0][..., 1]

# Denormalize
vp = vp * std_vp + mean_vp
vs = vs * std_vs + mean_vs

# From gpu to cpu and numpy
vp = vp.cpu().detach().numpy()[npml:-npml, npml:-npml]
vs = vs.cpu().detach().numpy()[npml:-npml, npml:-npml]

fig, axes=plt.subplots(1,2,figsize=(7,2))
axes[0].imshow(vp, cmap="seismic", vmin=1500, vmax=2500, aspect="auto")
axes[0].set_title("Vp")
axes[1].imshow(vs, cmap="seismic", vmin=1500/vp_vs_ratio, vmax=2500/vp_vs_ratio, aspect="auto")
axes[1].set_title("Vs")
plt.tight_layout()
plt.show()

fig,axes=plt.subplots(2,1,figsize=(6,4))
axes[0].plot(vp[:,32], label='inverted')
axes[0].plot(true_vp[:,32], label='true')
axes[1].plot(vs[:,96], label='inverted')
axes[1].plot(true_vs[:,96], label='true')
axes[0].set_title("Vp")
axes[1].set_title("Vs") 
plt.tight_layout()
plt.plot()


