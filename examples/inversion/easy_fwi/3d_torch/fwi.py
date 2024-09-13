import torch
import time
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from configures import *

import os
os.makedirs("figures", exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training
criterion = torch.nn.MSELoss()
lr = 10.
epochs = 101

# Load velocity
vel = np.load("models/true_vp.npy")
init = np.load("models/init_vp.npy")
# padding with pml
vel = np.pad(vel, ((pmln, pmln),)*vel.ndim, mode="edge")
init = np.pad(init, ((pmln, pmln),)*vel.ndim, mode="edge")
pmlc = generate_pml_coefficients_3d(vel.shape, N=pmln, multiple=False)

domain = vel.shape
nz, ny, nx = domain
# imshow(vel, vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 4))

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
amp = amp[freqs <= 20] 
freqs = freqs[freqs <= 20]
plt.plot(freqs, amp)
plt.title("Frequency spectrum")
plt.show()

# Geometry
src_loc = [[nx//2, ny//2, srcz]]

# cross-line
step = 4
receiver_locx = np.arange(pmln, nx-pmln, step)
receiver_locy = np.ones_like(receiver_locx)*(ny//2)
receiver_locz = np.ones_like(receiver_locx)*recz
# in-line
recxs = np.concatenate((receiver_locx, np.ones_like(np.arange(pmln, nx-pmln, step))*(nx//2)))
recys = np.concatenate((receiver_locy, np.arange(pmln, nx-pmln, step)))
reczs = np.concatenate((receiver_locz, np.ones_like(np.arange(pmln, nx-pmln, step))*recz))

rec_loc = [[recxs.tolist(), recys.tolist(), reczs.tolist()]]

# show geometry
plt.imshow(vel[0,:,:], cmap="seismic", aspect='auto', extent=[0, nx, ny, 0])

plt.scatter([src[0] for src in src_loc], [src[1] for src in src_loc], 
            c="r", marker="v", label="Sources")
plt.scatter(rec_loc[0][0], rec_loc[0][1], s=4, c="b", marker="^", 
            label="Receivers")
plt.legend()
plt.xlabel("x (grid)")
plt.ylabel("y (grid)")
plt.title("xoy pline")
plt.savefig("model_geometry.png", dpi=300)
plt.show()

# forward for observed data
# To GPU
vel = torch.from_numpy(vel).float().to(dev)
start_time = time.time()
with torch.no_grad():
    rec_obs = forward(wave, vel, pmlc, np.array(src_loc), np.array(rec_loc), domain, dt, dh, dev, recz, pmln)
end_time = time.time()
print(f"Forward modeling time: {end_time - start_time:.2f}s")
# Show gathers
vmin,vmax=np.percentile(rec_obs.cpu().numpy(), [2, 98])
plt.imshow(rec_obs.cpu().numpy()[0], vmin=vmin, vmax=vmax, cmap="seismic", aspect='auto')
plt.colorbar()
plt.title("Observed data")
plt.show()
# forward for initial data
# To GPU
init = torch.from_numpy(init).float().to(dev)
init.requires_grad = True
# Configures for training
opt = torch.optim.Adam([init], lr=lr)

def closure():
    opt.zero_grad()
    rand_size = 1
    rand_shots = np.random.randint(0, len(src_loc), size=rand_size).tolist()
    rec_syn = forward(wave, init, pmlc, np.array(src_loc), np.array(rec_loc), domain, dt, dh, dev, recz, pmln)
    loss = criterion(rec_syn, rec_obs[rand_shots])
    loss.backward()
    return loss
Loss = []
for epoch in tqdm.trange(epochs):
    loss = opt.step(closure)
    Loss.append(loss.item())
    if epoch % 10 == 0:
        # show gradient
        grad = init.grad.cpu().detach().numpy()[pmln:-pmln,pmln:-pmln,pmln:-pmln]
        vmin,vmax=np.percentile(grad, [2, 98])
        imshow(grad[:, :, nx//2], vmin=vmin, vmax=vmax, cmap="seismic", figsize=(5, 3))
        plt.show()
        # show vel
        show_data = init.cpu().detach().numpy()[pmln:-pmln,pmln:-pmln,pmln:-pmln]
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        imshow(show_data[:, :, nx//2], vmin=1500, vmax=2000, cmap="seismic", figsize=(5, 3), savepath=f"figures/{epoch:03d}.png")
        plt.show()
        # show slice
        plt.plot(show_data[:, ny//2, nx//2], label="Inverted model")
        plt.plot(vel.cpu().numpy()[pmln:-pmln,pmln:-pmln,pmln:-pmln][:, ny//2, nx//2], label="True model")
        plt.legend()
        plt.show()
        # Show loss
        plt.plot(Loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()