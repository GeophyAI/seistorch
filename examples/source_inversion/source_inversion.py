import sys, os
sys.path.append("../..")
import torch, tqdm, lesio
import numpy as np
import matplotlib.pyplot as plt
from seistorch.equations.acoustic import _time_step
from seistorch.model import build_model
from seistorch.setup import setup_src_rec, setup_rec_coords, setup_src_coords
from seistorch.utils import ricker_wave

from yaml import load
from yaml import CLoader as Loader

# Load obs data
# Only use the first shot gather
nt = 800 # number of time steps used for source inversion
obs = np.load("./shot_gather.npy", allow_pickle=True)[0][:nt]

config_path = r"configs/source_inversion.yml"
save_path = r"./results"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)


os.makedirs(save_path, exist_ok=True)

"""show the obs data"""
vmin, vmax=np.percentile(obs, [2, 98])
kwargs = {"vmin":vmin, "vmax":vmax, "cmap":"seismic", "aspect":"auto"}
plt.imshow(obs, **kwargs)
plt.show()
plt.savefig(f"{save_path}/obs.png")

# Setup the initial wavelet
true = np.load("wavelet_bspline.npy")
init = ricker_wave(5, cfg['geom']['dt'], nt, cfg['geom']['wavelet_delay']*2, dtype="numpy")

"""show the true and initial wavelet"""
fig, ax=plt.subplots(1,1,figsize=(8,4))
ax.plot(true, label="True")
ax.plot(init, label="Initial")
ax.legend()
ax.set_title("True and Initial Wavelet")
plt.savefig(f"{save_path}/true_initial_wavelet.png")
plt.show()


# Setup model
cfg, model = build_model(config_path, mode="inversion")
print(f"Target devide {cfg['device']}")
# Setup wavelet
inverted = torch.from_numpy(init).unsqueeze(0).to(cfg['device'])
inverted.requires_grad = True

# Load the source and receiver coordinates
src_list, rec_list, full_rec_list, fixed_receivers = setup_src_rec(cfg)
probes = setup_rec_coords(full_rec_list, cfg['geom']['pml']['N'])
# Move model to device
model.to(cfg['device'])
# Setup receivers
model.reset_probes(probes)
# Setup sources
src = setup_src_coords(src_list[0], cfg['geom']['pml']['N'])
model.reset_sources([src])
# Setup optimizer and loss function
optimizer = torch.optim.Adam([inverted], lr=0.01)
critic = torch.nn.MSELoss()

Epochs = 201
pbar = tqdm.trange(Epochs)
LOSS = []
for epoch in pbar:
    optimizer.zero_grad()
    # Forward modeling
    syn = model(inverted)
#     if epoch==0: init_data = ypred.cpu().detach().numpy()
#     # Calculate the loss
    loss = critic(syn, torch.from_numpy(obs).to(cfg['device']))
    loss.backward()
    optimizer.step()
    pbar.set_description(f"Loss:{loss}")
    LOSS.append(loss.cpu().detach().numpy())

    if epoch%20==0:
        fig, axes = plt.subplots(5,1, figsize=(10,8))
        axes[0].plot(inverted.cpu().detach().numpy()[0], c='r', label='Inverted')
        axes[0].plot(init, c='b', label='Initial')
        axes[0].plot(true, c='g', label='True')
        axes[0].set_title("Inverted source wavelet")
        axes[0].legend(ncol=3)

        # # Plot the forward modeling result
        vmin, vmax=np.percentile(obs, [2, 98])
        kwargs = {"vmin":vmin, "vmax":vmax, "cmap":"seismic", "aspect":"auto"}
        axes[1].imshow(syn.cpu().detach().numpy()[:,0:200], **kwargs)
        axes[1].set_title("Predicted data")
        axes[2].imshow(obs, **kwargs)
        axes[2].set_title("Observed data")

        # show the loss
        axes[3].plot(LOSS)
        axes[3].set_title("Loss")
        
        plt.tight_layout()
        plt.savefig(f"results/Results{epoch:03d}.png")
        plt.close()

# Plot the fina result
fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.plot(inverted.cpu().detach().numpy()[0], c='r', label='Inverted')
ax.plot(init, c='b', label='Initial')
ax.plot(true, c='g', label='True')
ax.legend(ncol=3)
ax.set_title("Inverted source wavelet")
plt.savefig(f"{save_path}/Final_result.png")