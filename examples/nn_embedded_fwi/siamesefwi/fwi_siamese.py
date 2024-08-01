import torch, tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from configure import *

torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
# Load velocity
vel = np.load("../../models/marmousi_model/linear_vp.npy")
vel = vel[:, expand:-expand][::model_scale, ::model_scale]
vel = np.pad(vel, ((pmln, pmln), (pmln, pmln)), mode="edge")
pmlc = generate_pml_coefficients_2d(vel.shape, N=pmln, multiple=False)
vel = torch.from_numpy(vel).float().to(dev)
vel.requires_grad = True
domain = vel.shape
nz, nx = domain
# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
# network
siamese = SiameseCNN().to(dev)
# Loss func
l2loss = torch.nn.MSELoss()
# Optimizer
opt = torch.optim.Adam([{'params': [vel], 'lr': lr_vel},
    {'params': siamese.parameters(), 'lr': lr_cnn}])
# Geometry
src_x = np.arange(pmln, nx-pmln, srcx_step)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]
# load observed data
obs = np.load("obs.npy")
obs = torch.from_numpy(obs).float().to(dev)
LOSS = []
kwargs = dict(wave=wave, b=pmlc, src_list = np.array(sources), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, pmln=pmln)
kwargs_imshow = dict(vmin=vmin, vmax=vmax, aspect='auto', cmap='seismic', extent=[0, (nx-2*pmln)*dh, (nz-2*pmln)*dh, 0])
for epoch in tqdm.trange(EPOCHS):
    # Select part of the data
    rand_shots = np.random.randint(0, len(sources), size=batch_size).tolist()
    kwargs.update(dict(src_list=np.array(sources)[rand_shots]))
    syn = forward(c=vel, **kwargs)
    # Loss
    _obs = obs[rand_shots].unsqueeze(1)
    _syn = syn.unsqueeze(1)
    latent_obs = siamese(_obs)
    latent_syn = siamese(_syn)
    loss = l2loss(latent_obs, latent_syn)
    opt.zero_grad()
    loss.backward()
    vel.grad[:water_grid] = 0.
    opt.step()
    LOSS.append(loss.item())

    if epoch % show_every == 0: 

        latent_obs = latent_obs.detach().cpu().numpy().squeeze()
        latent_syn = latent_syn.detach().cpu().numpy().squeeze()

        # If number of feature map is 1
        fig, axes = plt.subplots(1, 4, figsize=(10, 6))
        for d, ax in zip([_obs[0].detach().cpu().numpy(), _syn[0].detach().cpu().numpy(), 
                          latent_obs[0], latent_syn[0]], axes):
            d = d.squeeze()
            vmin, vmax = np.percentile(d, [1, 99])
            ax.imshow(d, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        plt.show()

        # fm_counts = latent_obs.shape[1]
        # cols = rows = np.sqrt(fm_counts) 
        # assert cols == int(cols)
        # cols = int(cols)
        # rows = int(rows)
        # fit, axes = plt.subplots(cols, rows, figsize=(10, 8))
        # for j, ax in enumerate(axes.ravel()):
        #     vmin, vmax = np.percentile(latent_obs[0][j], [1, 99])
        #     ax.imshow(latent_obs[0][j], vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        #     ax.set_title(f"Obs Feature map {j}")
        #     ax.axis("off")
        # plt.tight_layout()
        # plt.show()

        # fm_counts = latent_obs.shape[1]
        # cols = rows = np.sqrt(fm_counts) 
        # assert cols == int(cols)
        # cols = int(cols)
        # rows = int(rows)
        # fit, axes = plt.subplots(cols, rows, figsize=(10, 8))
        # for j, ax in enumerate(axes.ravel()):
        #     vmin, vmax = np.percentile(latent_syn[0][j], [1, 99])
        #     ax.imshow(latent_syn[0][j], vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        #     ax.set_title(f"Syn Feature map {j}")
        #     ax.axis("off")
        # plt.tight_layout()
        # plt.show()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        inverted = vel.cpu().detach().numpy()[pmln:-pmln, pmln:-pmln]
        plt.imshow(inverted, **kwargs_imshow)
        plt.show()

plt.plot(LOSS)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

torch.save(siamese, "siamese.pth")
np.save("inverted_by_siamese.npy", vel.cpu().detach().numpy()[pmln:-pmln, pmln:-pmln])

