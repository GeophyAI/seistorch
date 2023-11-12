import sys, torch
sys.path.append("../../../..")
import numpy as np
import matplotlib.pyplot as plt
from seistorch.model import build_model
from seistorch.io import SeisIO
from seistorch.setup import setup_src_coords, setup_rec_coords
from seistorch.signal import ricker_wave
from seistorch.loss import L2, Sinkhorn, Wasserstein1d

"""
Configures
"""
device = "cuda" # or "cpu"
config_path= "./forward.yml"
shot_no = 5

# Load the configure file
io = SeisIO(load_cfg=False)
cfg = io.read_cfg(config_path)

# build the model
_, model = build_model(config_path, device="cuda", mode="inversion")
obs = np.load(cfg['geom']['obsPath'], allow_pickle=True)[shot_no]

obs = torch.from_numpy(obs).to(device).unsqueeze(0)

# read acquisition geometry
srcs = io.read_pkl(cfg['geom']['sources']) # list of source coordinates
recs = io.read_pkl(cfg['geom']['receivers']) # list of receiver coordinates

# pad the source and receiver coordinates
srcs = setup_src_coords(srcs[shot_no], cfg['geom']['pml']['N'], multiple=False)
recs = setup_rec_coords(recs[shot_no], cfg['geom']['pml']['N'], multiple=False)

# set the source and receiver coordinates
model.reset_sources(srcs)
model.reset_probes(recs)

model.to(device)
model.train()
# set the wavelet data
wavelet = ricker_wave(fm=cfg['geom']['fm'], 
                      dt=cfg['geom']['dt'], 
                      nt=cfg['geom']['nt'], 
                      delay=cfg['geom']['wavelet_delay'], dtype='tensor', inverse=False)
wavelet=wavelet.to(device).unsqueeze(0)

# different loss functions
l2loss = L2()
otloss = Wasserstein1d()#Sinkhorn()

# Forward modeling
syn = model(wavelet)
print("Calculating the adjoint source of l2 loss function...")
adj_l2 = torch.autograd.grad(l2loss(syn.stack(), obs), 
                             syn, 
                             create_graph=True, 
                             retain_graph=False)[0].detach().cpu().numpy()
print("Calculating the adjoint source of OT loss function...")
adj_ot = torch.autograd.grad(otloss(syn.stack(), obs), 
                             syn, 
                             create_graph=True, 
                             retain_graph=False)[0].detach().cpu().numpy()


# Show the results
dt = cfg['geom']['dt']
dh = cfg['geom']['h']
titles = ["Synthetic", "Observed", "L2 Adjoint", "OT Adjoint"]
extent = [0, obs.shape[2]*dh, obs.shape[1]*dt, 0]
fig, axes=plt.subplots(1,4,figsize=(12,4))
for d, ax, title in zip([syn.numpy()[0], obs.cpu().numpy(), adj_l2, adj_ot], axes.ravel(), titles):
    d = d.squeeze()
    vmin, vmax = np.percentile(d, [1, 99])
    kwargs=dict(aspect='auto', vmin=vmin, vmax=vmax, cmap=plt.cm.gray, extent=extent)
    _ax=ax.imshow(d, **kwargs)
    ax.set_title(title)
    plt.colorbar(_ax, ax=ax)
plt.tight_layout()
plt.show()
fig.savefig("adjoint_source.png")







