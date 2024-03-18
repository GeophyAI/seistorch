import sys, torch
sys.path.append("../../..")
import numpy as np
import matplotlib.pyplot as plt
from seistorch.model import build_model
from seistorch.io import SeisIO
from seistorch.setup import setup_src_coords, setup_rec_coords
from seistorch.signal import ricker_wave
from seistorch.loss import *
import lesio

"""
Configures
"""
device = "cuda" # or "cpu"
config_path= "./observed.yml"
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
l1 = L1()
ig = Integration()
phase = Phase()
sinkhorn = Sinkhorn()
l2loss = L2()
otloss = Wasserstein1d()#Sinkhorn()
nim = NormalizedIntegrationMethod()
env = Envelope()

losses = [l2loss, otloss, ig, env]

# Forward modeling
syn = model(wavelet)
adjs = []
for loss in losses:
    print(f"Calculating the adjoint source of {loss} loss function...")

    adj = torch.autograd.grad(loss(syn.stack(), obs), 
                              syn, 
                              create_graph=True, 
                              retain_graph=False)[0].detach().cpu().numpy()
    adjs.append(adj)

# Show the results
dt = cfg['geom']['dt']
dh = cfg['geom']['h']
titles = ["Synthetic", "Observed", *[loss.name for loss in losses]]
extent = [0, obs.shape[2]*dh, obs.shape[1]*dt, 0]
fig, axes=plt.subplots(2,3,figsize=(8,6))
for d, ax, title in zip([syn.numpy()[0], 
                         obs.cpu().numpy(), 
                         *adjs], axes.ravel(), titles):
    d = d.squeeze()
    vmin, vmax = np.percentile(d, [1, 99])
    kwargs=dict(aspect='auto', vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, extent=extent)
    _ax=ax.imshow(d, **kwargs)
    ax.set_title(title)
    plt.colorbar(_ax, ax=ax)
plt.tight_layout()
plt.show()
fig.savefig("adjoint_source.png")

# Trace
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(syn.numpy()[0][:,100,:], label="Synthetic")
ax.plot(obs.cpu().numpy()[0,:,100], label="Observed")
for d, loss in zip(adjs, losses):
    ax.plot(d[:,100], label=loss.name)
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig("adjoint_trace.png")

# Freq
fig, ax = plt.subplots(figsize=(6,4))
titles = ["Synthetic", "Observed", *[loss.name for loss in losses]]
for d, title in zip([syn.numpy()[0], obs.cpu().numpy()[0], *adjs], titles):
    amp, freq = lesio.tools.freq_spectrum(d)
    ax.plot(freq[0:48], amp[0:48]/amp.max(), label=title)
    ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6,4))
for d, loss in zip(adjs, losses):
    if loss.name =='w1d':
        ax.plot(d[:,100], label=loss.name)
ax.legend()
plt.tight_layout()
plt.show()







