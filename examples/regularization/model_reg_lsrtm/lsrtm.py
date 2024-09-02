import sys, torch, os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from torchvision.transforms import Pad

torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
from tv_bregman import denoise_tv_bregman

from configures import *
torch.manual_seed(seed)
np.random.seed(seed)

from utils import *

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def regularization_model(m, x=True, z=True):
    
    l=0.

    # TV
    grad_x = m[:,1:]-m[:,:-1]
    grad_z = m[1:]-m[:-1]

    grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
    grad_z = torch.nn.functional.pad(grad_z, (0, 0, 0, 1))

    l += torch.sqrt(grad_x**2 + grad_z**2 + 1e-16)

    return l.sum()/torch.prod(torch.tensor(m.shape))

# Load velocity
smvp = np.load("models/smooth_vp.npy")
true_m = np.load("models/true_reflectivity.npy")
zero_m = np.zeros_like(true_m)
os.makedirs(save_path, exist_ok=True)
# Padding
padding = Pad((bwidth, bwidth, bwidth, bwidth), padding_mode='edge')

# Get the shape of the model
nz, nx = smvp.shape
domain = (nz+2*bwidth, nx+2*bwidth)

# HABC coefficients
pmlc = generate_pml_coefficients_2d(domain, N=bwidth).to(dev)

# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
tt = np.arange(nt) * dt
plt.plot(tt, wave.cpu().numpy())
plt.title("Wavelet")
plt.show()
# Frequency spectrum
# Show freq < 10Hz
freqs = np.fft.fftfreq(nt, dt)[:nt//2]
amp = np.abs(np.fft.fft(wave.cpu().numpy()))[:nt//2]
amp = amp[freqs <= 30] 
freqs = freqs[freqs <= 30]
plt.plot(freqs, amp)
plt.title("Frequency spectrum")
plt.show()

# Geometry
srcxs = np.arange(bwidth, domain[1]-bwidth, srcx_step).tolist()
srczs = (np.ones_like(srcxs) * srcz).tolist()
src_loc = list(zip(srcxs, srczs))

recxs = np.arange(bwidth, domain[1]-bwidth, 1).tolist()
reczs = (np.ones_like(recxs) * recz).tolist()
rec_loc = list(zip(recxs, reczs))


# forward for observed data
# To GPU
smvp = torch.from_numpy(smvp).float().to(dev)
zero_m = torch.from_numpy(zero_m).float().to(dev)
zero_m = padding(zero_m)
zero_m.requires_grad = True
smvp.requires_grad = False # default is False

kwargs = dict(b=pmlc, src_list=np.array(src_loc), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, bwidth=bwidth)

# load data
rec_born = np.load("born.npy")
rec_born = torch.from_numpy(rec_born).float().to(dev)

opt = torch.optim.Adam([zero_m], lr=lr)
criterion = torch.nn.MSELoss()

def closure(vp, m, use_reg=True):
    opt.zero_grad()
    rand_shots = np.random.randint(0, len(src_loc), size=batch_size).tolist()
    kwargs = dict(b=pmlc, src_list=np.array(src_loc)[rand_shots], domain=domain, dt=dt, h=dh, dev=dev, recz=recz, bwidth=bwidth)
    rec_syn = forward(wave, m, vp, **kwargs)
    loss = criterion(rec_syn, rec_born[rand_shots])
    if use_reg:
        loss += 1e-2 * regularization_model(m, x=True, z=True)
    loss.backward()
    return loss

Loss = []
for epoch in tqdm.trange(epochs):

    # ref = padding(zero_m)
    ref = zero_m
    vp = padding(smvp)
    loss = opt.step(partial(closure, m=ref, vp=vp, use_reg=False))

    Loss.append(loss.item())
    if epoch % 10 == 0:
        
        inverted = zero_m.cpu().detach().numpy()[bwidth:-bwidth, bwidth:-bwidth]
        # denoise_inv = denoise_tv_bregman(zero_m, 0.1, max_iter=100, eps=1e-3)

        # fig, axes= plt.subplots(1, 2, figsize=(8, 4))
        # axes[0].imshow(inverted, vmin=-0.5, vmax=0.5, cmap="gray", aspect="auto")
        # axes[0].set_title("Inverted")
        # axes[1].imshow(denoise_inv.cpu().detach().numpy()[bwidth:-bwidth, bwidth:-bwidth], vmin=-0.5, vmax=0.5, cmap="gray", aspect="auto")
        # axes[1].set_title("Denoised")
        # plt.tight_layout()
        # plt.show()

        fig, axes= plt.subplots(3, 1, figsize=(6, 10))
        extent = [0, nx*dh, nz*dh, 0]
        axes[0].imshow(inverted, vmin=-0.5, vmax=0.5, extent=extent, cmap="gray", aspect="auto")

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        axes[1].plot(true_m[:,nx//2], label="True")
        axes[1].plot(inverted[:,nx//2], label="Inverted")
        axes[1].legend()

        axes[2].plot(Loss)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{epoch:02d}.png"))
        plt.show()
