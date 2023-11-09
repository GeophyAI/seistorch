import sys, torch
sys.path.append("../../")
import numpy as np
import matplotlib.pyplot as plt
from seistorch.model import build_model
from seistorch.io import SeisIO
from seistorch.setup import setup_src_coords, setup_rec_coords
from seistorch.signal import ricker_wave
from seistorch.loss import L2

"""
Configures
"""
device = "cuda" # or "cpu"
config_path= "./config/implicit.yml"


# Load the configure file
io = SeisIO(load_cfg=False)
cfg = io.read_cfg(config_path)

# build the model
_, model = build_model(config_path, device="cuda", mode="inversion")
obs = np.load(cfg['geom']['obsPath'], allow_pickle=True)[0]
print(obs.shape)
obs = torch.from_numpy(obs).to(device).unsqueeze(0)

# read acquisition geometry
srcs = io.read_pkl(cfg['geom']['sources']) # list of source coordinates
recs = io.read_pkl(cfg['geom']['receivers']) # list of receiver coordinates

# pad the source and receiver coordinates
srcs = setup_src_coords(srcs[0], cfg['geom']['pml']['N'], multiple=False)
recs = setup_rec_coords(recs[0], cfg['geom']['pml']['N'], multiple=False)

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
model.cell.geom.step(None)
# different loss functions
l2loss = torch.nn.MSELoss()

# Forward modeling
syn = model(wavelet)
print("Calculating the adjoint source of l2 loss function...")
adj_l2 = torch.autograd.grad(l2loss(syn, obs), 
                             syn, 
                             create_graph=True, 
                             retain_graph=False)[0]

adj_byhand = 2 * (syn - obs) / syn.numel()

print(torch.allclose(adj_l2, adj_byhand))

# print("Calculating the adjoint source of l1 loss function...")
# adj_l1 = torch.autograd.grad(l1loss(syn, obs), 
#                              syn, 
#                              create_graph=True, 
#                              retain_graph=False)[0]
# print("Calculating the adjoint source of envelope loss function...")
# adj_en = torch.autograd.grad(envloss(syn, obs), 
#                              syn, 
#                              create_graph=True, 
#                              retain_graph=False)[0]


# Show the results
dt = cfg['geom']['dt']
dh = cfg['geom']['h']
titles = ["Synthetic", "Observed", "L2 Adjoint by torch", "L2 Adjoint by hand"]
extent = [0, obs.shape[2]*dh, obs.shape[1]*dt, 0]
fig, axes=plt.subplots(1,4,figsize=(12,4))
for d, ax, title in zip([syn, obs, adj_l2, adj_byhand], axes.ravel(), titles):
    d = d.detach().cpu().numpy().squeeze()
    vmin, vmax = np.percentile(d, [1, 99])
    kwargs=dict(aspect='auto', vmin=vmin, vmax=vmax, cmap=plt.cm.gray, extent=extent)
    _ax=ax.imshow(d, **kwargs)
    ax.set_title(title)
    plt.colorbar(_ax, ax=ax)
plt.tight_layout()
plt.show()
fig.savefig("adjoint_source.png")







