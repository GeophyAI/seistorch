import torch, tqdm, os
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
from utils import *
from configure2 import *
torch.manual_seed(seed)
import matplotlib.pyplot as plt
from siren import Siren, MLP
from torch.optim import lr_scheduler

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Load rho
rho = np.load("models/rho.npy")
rho = np.pad(rho, ((npml, npml), (npml, npml)), mode='edge')
rho = to_tensor(rho, dev)

# Load observed data
rec_obs = np.load("obs.npy")
rec_obs = to_tensor(rec_obs, dev)

# Load wavelet 
wavelet = ricker(np.arange(nt) * dt-delay*dt, f=fm)
fig,ax=plt.subplots(1,1,figsize=(5,4))
ax.plot(np.arange(nt)*dt, wavelet)
ax.set_title('Ricker Wavelet')
ax.set_xlabel('Time (s)')
plt.show()
show_freq_spectrum(wavelet.reshape(nt,1,1), dt=dt, end_freq=25, title='Frequency Spectrum')

# Load velocity
domain = (nz+2*npml, nx+2*npml)
pmlc = generate_pml_coefficients_2d(domain, npml)

# Load model
imvp = Siren(in_features=in_features,
             out_features=out_features, 
             hidden_features=hidden_features, 
             hidden_layers=hidden_layers, 
             outermost_linear=True,
             domain_shape=domain).to(dev)
imvs = Siren(in_features=in_features,
             out_features=out_features, 
             hidden_features=hidden_features, 
             hidden_layers=hidden_layers, 
             outermost_linear=True,
             domain_shape=domain).to(dev)
# imvp = MLP(in_features=in_features,
#            out_features=out_features, 
#            hidden_features=hidden_features, 
#            hidden_layers=hidden_layers, 
#            domain_shape=domain).to(dev)
# imvs = MLP(in_features=in_features,
#            out_features=out_features, 
#            hidden_features=hidden_features, 
#            hidden_layers=hidden_layers, 
#            domain_shape=domain).to(dev)
# Transfer to tensor
wavelet = wavelet.to(dev)
pmlc = pmlc.to(dev)

# Geometry
src_x = np.arange(npml, nx+npml, 10)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]
kwargs = dict(wave=wavelet, pmlc=pmlc, src_list=sources, domain=domain, dt=dt, h=dh, dev=dev, recz=recz, npml=npml)

opt_vp = torch.optim.Adam(imvp.parameters(), lr=lr)
opt_vs = torch.optim.Adam(imvs.parameters(), lr=lr)

l2loss = torch.nn.MSELoss()
scheduler_vp = lr_scheduler.ExponentialLR(opt_vp, lr_decay)
scheduler_vs = lr_scheduler.ExponentialLR(opt_vs, lr_decay)
coords = imvp.coords.to(dev)

std_vs, mean_vs = std_vp/vp_vs_ratio, mean_vp/vp_vs_ratio

# Run Implicit inversion simulation
# forward for predicted data
LOSS = []
for epoch in tqdm.trange(EPOCHS):

    vp = imvp(coords)[0][..., 0]
    vs = imvs(coords)[0][..., 0]

    # Denormalize
    vp = vp * std_vp + mean_vp
    vs = vs * std_vs + mean_vs

    # forward
    kwargs.update(parameters=[vp, vs, rho])
    rec_pred = forward(**kwargs)
    loss = l2loss(rec_pred, rec_obs)
    LOSS.append(loss.item())
    opt_vp.zero_grad()
    opt_vs.zero_grad()
    loss.backward()

    opt_vp.step()
    opt_vs.step()

    scheduler_vp.step()
    scheduler_vs.step()

    if epoch % show_every == 0:
        print(vp.shape)
        fig, axes=plt.subplots(1, 2, figsize=(7, 2))
        vp = vp.detach().cpu().numpy()[npml:-npml, npml:-npml]
        vs = vs.detach().cpu().numpy()[npml:-npml, npml:-npml]
        axes[0].imshow(vp, cmap="seismic", vmin=1500, vmax=2500, aspect="auto")
        axes[0].set_title("Vp")
        axes[1].imshow(vs, cmap="seismic", vmin=1500/vp_vs_ratio, vmax=2500/vp_vs_ratio, aspect="auto")
        axes[1].set_title("Vs")
        plt.tight_layout()
        plt.savefig(f'figures/epoch_{epoch:04d}.png')
        plt.show()
        torch.save(imvp, f'results/model_vp_{epoch:04d}.pt')
        torch.save(imvs, f'results/model_vs_{epoch:04d}.pt')
        fig,ax=plt.subplots(1,1,figsize=(5,3))
        ax.plot(LOSS)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        plt.yscale("log")
        plt.tight_layout()
        plt.show()