import torch, tqdm, os
import numpy as np
from utils import *
from configure import *
torch.manual_seed(seed)
import matplotlib.pyplot as plt
from networks import FWINET
from torch.optim import lr_scheduler
from torchvision.transforms import Pad

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Load rho
# rho = np.load("models/rho.npy")
# rho = np.pad(rho, ((npml, npml), (npml, npml)), mode='edge')
# rho = to_tensor(rho, dev)

# Load observed data
rec_obs = np.load("obs.npy")
rec_obs = to_tensor(rec_obs, dev) # with shape (shot_counts, 2, nsamples, nrecs)
rec_obs = rec_obs.permute(1, 0, 2, 3) # with shape (2, shot_counts, nsamples, nrecs)

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
_, shot_counts, nsamples, nrecs = rec_obs.shape
net = FWINET(batch_size, nsamples, nrecs).to(dev)

init_vp = to_tensor(np.load("models/bg_vp.npy"), dev)
init_vs = to_tensor(np.load("models/bg_vs.npy"), dev)
init_rho = to_tensor(np.load("models/bg_rho.npy"), dev)
print(init_vp.mean(), init_vs.mean(), init_rho.mean())
# Transfer to tensor
wavelet = wavelet.to(dev)
pmlc = pmlc.to(dev)

# Geometry
src_x = np.arange(npml, nx+npml, src_x_step)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]
kwargs = dict(wave=wavelet, pmlc=pmlc, src_list=sources, domain=domain, dt=dt, h=dh, dev=dev, recz=recz, npml=npml)

opt = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(opt, lr_decay)
l2loss = torch.nn.MSELoss()

padding_pml = Pad((npml, npml, npml, npml), padding_mode='edge')
# Run Implicit inversion simulation
# forward for predicted data
LOSS = []
for epoch in tqdm.trange(EPOCHS):

    # Randomly select shots
    rand_shots = np.random.choice(shot_counts, batch_size, replace=False)

    # forward
    delta_vp, delta_vs, delta_rho = net(rec_obs[0:1][:,rand_shots], 
                                        rec_obs[1:2][:,rand_shots])

    vp = delta_vp*100+init_vp
    vs = delta_vs*100+init_vs
    rho = delta_rho*100+init_rho

    vp = padding_pml(vp)
    vs = padding_pml(vs)
    rho = padding_pml(rho)

    # The following 3 lines are Very vEry veRy verY important
    vp = torch.clamp(vp, min=background_vp, max=anaomaly_vp)
    vs = torch.clamp(vs, min=background_vp/vp_vs_ratio, max=anaomaly_vp/vp_vs_ratio)
    rho = torch.clamp(rho, min=background_rho, max=anaomaly_rho)

    kwargs.update(parameters=[vp, vs, rho])
    kwargs.update(src_list=np.array(sources)[rand_shots])
    rec_pred = forward(**kwargs)
    loss = l2loss(rec_pred.permute(1, 0, 2, 3), rec_obs[:,rand_shots])
    LOSS.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()

    if epoch % show_every == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        fig, axes=plt.subplots(1,3,figsize=(10,3))
        vp = vp.detach().cpu().numpy().squeeze()[npml:-npml, npml:-npml]
        vs = vs.detach().cpu().numpy().squeeze()[npml:-npml, npml:-npml]
        rho = rho.detach().cpu().numpy().squeeze()[npml:-npml, npml:-npml]

        plt.colorbar(axes[0].imshow(vp, cmap="seismic", vmin=background_vp, vmax=anaomaly_vp, aspect="auto"), orientation='horizontal')
        axes[0].set_title("Vp")

        plt.colorbar(axes[1].imshow(vs, cmap="seismic", vmin=background_vp/vp_vs_ratio, vmax=anaomaly_vp/vp_vs_ratio, aspect="auto"), orientation='horizontal')
        axes[1].set_title("Vs")

        plt.colorbar(axes[2].imshow(rho, cmap="seismic", vmin=background_rho, vmax=anaomaly_rho, aspect="auto"), orientation='horizontal')
        axes[2].set_title("Rho")
        plt.tight_layout()
        plt.savefig(f"figures/epoch_{epoch}.png")
        plt.show()

        # Show loss
        plt.figure(figsize=(5,3))
        plt.plot(LOSS)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.savefig("figures/loss.png")
        plt.show()