import torch, os, tqdm
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
from configure import *
torch.random.manual_seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from utils_torch import forward, ricker, generate_pml_coefficients_2d
from networks import FWINET

from torchvision.transforms import Pad
from torch.optim.lr_scheduler import ExponentialLR

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l2loss = torch.nn.MSELoss()
domain = (nz+2*npml, nx+2*npml)

# load observed data
obs = np.load("obs.npy")
obs = torch.from_numpy(obs).float().to(dev)
obs = obs.unsqueeze(0)
# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)

# load true model for comparison
true = np.load("models/vp.npy")

# PML coefficients
pmlc = generate_pml_coefficients_2d(domain, npml)
pmlc = pmlc.to(dev)
padding_pml = Pad((npml, npml, npml, npml), padding_mode='edge')

# show spectrum of wavelet
plt.figure(figsize=(5, 3))
amp = np.abs(np.fft.fft(wave.cpu().numpy()))
freqs = np.fft.fftfreq(nt, dt)
amp = amp[freqs >= 0]
freqs = freqs[freqs >= 0]
plt.plot(freqs[:50], amp[:50])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Geometry
src_x = np.linspace(npml, nx+npml, 20)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(npml, nx+npml, 1)
receiver_locz = np.ones_like(receiver_locx)*recz

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

# implicit neural network for representing velocity
_, shots, nsamples, nrecs = obs.shape
model = FWINET(shots, nsamples, nrecs, min_filters=min_filters, latent_length=latent_length).to(dev)
# count parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
opt = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ExponentialLR(opt, lr_decay)
kwargs = dict(wave=wave, src_list = np.array(sources), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, b=pmlc)
# forward for predicted data
LOSS = []
# Denormalization with wrong estimation
vp_min = background_vp
vp_max = anaomaly_vp
for epoch in tqdm.trange(EPOCHS):
    vp = model(obs)
    vp = vp_min + vp*(vp_max-vp_min)
    vp = padding_pml(vp)
    # forward
    syn = forward(c=vp, **kwargs)
    loss = l2loss(syn.unsqueeze(0), obs)
    LOSS.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()

    if epoch % show_every == 0:
        plt.figure(3, figsize=(6, 8))
        # show inverted
        ax = plt.subplot(311)
        inverted = vp.cpu().detach().numpy().reshape(domain)
        inverted = inverted[npml:-npml, npml:-npml]
        show_kwargs = dict(cmap="seismic", aspect="auto", vmin=background_vp, vmax=anaomaly_vp)
        ax.imshow(inverted, **show_kwargs)
        # show loss
        ax = plt.subplot(312)
        ax.plot(LOSS)
        plt.yscale("log")
        # show compare
        ax = plt.subplot(313)
        ax.plot(true[:,nx//2], 'r', label="True")
        ax.plot(inverted[:,nx//2], 'b', label="Inverted")
        ax.legend()
        plt.tight_layout()
        # plt.savefig(f"figures/{epoch:04d}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"figures/{epoch:04d}.png")
        plt.show()
        np.save(f"results/inverted{epoch:04d}.npy", inverted)



