import torch, tqdm, os
import numpy as np
from utils import *
from configure import *
torch.manual_seed(seed)
import matplotlib.pyplot as plt
from siren import Siren
from torch.optim import lr_scheduler

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

dev = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Load rho
# rho = np.load("models/rho.npy")
# rho = np.pad(rho, ((npml, npml), (npml, npml)), mode='edge')
# rho = to_tensor(rho, dev)

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
imrho = Siren(in_features=in_features, 
              out_features=out_features, 
              hidden_features=hidden_features, 
              hidden_layers=hidden_layers, 
              outermost_linear=True,
              domain_shape=domain).to(dev)

# Transfer to tensor
wavelet = wavelet.to(dev)
pmlc = pmlc.to(dev)

# Geometry
src_x = np.arange(npml, nx+npml, 10)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]
kwargs = dict(wave=wavelet, pmlc=pmlc, src_list=sources, domain=domain, dt=dt, h=dh, dev=dev, recz=recz, npml=npml)

opt = torch.optim.Adam([{'params': imvp.parameters()}, 
                        {'params': imvs.parameters()},
                        {'params': imrho.parameters()}], lr=lr)
scheduler = lr_scheduler.ExponentialLR(opt, lr_decay)
l2loss = torch.nn.MSELoss()
coords = imvp.coords.to(dev)

std_vs, mean_vs = std_vp/vp_vs_ratio, mean_vp/vp_vs_ratio

# Run Implicit inversion simulation
# forward for predicted data
LOSS = []
for epoch in tqdm.trange(EPOCHS):

    vp = imvp(coords)[0][..., 0]
    vs = imvs(coords)[0][..., 0]
    rho = imrho(coords)[0][..., 0]

    # Denormalize
    vp = vp * std_vp + mean_vp
    vs = vs * std_vs + mean_vs
    rho = rho * std_rho + mean_rho

    # forward
    kwargs.update(parameters=[vp, vs, rho])
    rec_pred = forward(**kwargs)
    loss = l2loss(rec_pred, rec_obs)
    LOSS.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()

    if epoch % show_every == 0:
        fig, axes=plt.subplots(1, 3, figsize=(10, 2))
        vp = vp.detach().cpu().numpy()[npml:-npml, npml:-npml]
        vs = vs.detach().cpu().numpy()[npml:-npml, npml:-npml]
        rho = rho.detach().cpu().numpy()[npml:-npml, npml:-npml]

        axes[0].imshow(vp, cmap="seismic", vmin=background_vp, vmax=anaomaly_vp, aspect="auto")
        axes[0].set_title("Vp")
        axes[1].imshow(rho, cmap="seismic", vmin=0.31*background_vp**0.25*1000, vmax=0.31*anaomaly_vp**0.25*1000, aspect="auto")
        axes[1].set_title("Rho")        
        axes[2].imshow(vs, cmap="seismic", vmin=background_vp/vp_vs_ratio, vmax=anaomaly_vp/vp_vs_ratio, aspect="auto")
        axes[2].set_title("Vs")
        plt.tight_layout()
        plt.savefig(f'figures/epoch_{epoch:04d}.png')
        plt.show()
        torch.save(imvp, f'results/model_vp_{epoch:04d}.pt')
        torch.save(imvs, f'results/model_vs_{epoch:04d}.pt')
        torch.save(imrho, f'results/model_rho_{epoch:04d}.pt')

        fig,ax=plt.subplots(1,1,figsize=(5,3))
        ax.plot(LOSS)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        plt.yscale("log")
        plt.tight_layout()
        plt.show()