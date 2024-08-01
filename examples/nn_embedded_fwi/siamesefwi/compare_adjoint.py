import torch, tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from configure import *

torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
batch_size = 2
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
for epoch in tqdm.trange(1):
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
    # Adjoint of Siamese FWI
    adj_siamese = torch.autograd.grad(loss, _syn, create_graph=True, retain_graph=True)[0]
    grad_siamese = torch.autograd.grad(loss, vel, create_graph=True, retain_graph=True)[0]
    # Adjoint of Classic FWI
    loss_classic = l2loss(_obs, _syn)
    adj_classic = torch.autograd.grad(loss_classic, _syn, create_graph=True, retain_graph=True)[0]
    grad_classic = torch.autograd.grad(loss_classic, vel, create_graph=True, retain_graph=True)[0]

# Show the adjoint sources
adj_siamese = adj_siamese.detach().cpu().numpy().squeeze()
adj_classic = adj_classic.detach().cpu().numpy().squeeze()

# Show the data
data = [_obs[0].detach().cpu().numpy(), 
        _syn[0].detach().cpu().numpy(),
        adj_siamese[0],
        adj_classic[0], 
        (latent_obs[0]-latent_syn[0]).detach().cpu().numpy()]

titles = ['Observed', 'Synthetic', 
          'Adjoint Siamese', 'Adjoint Classic', 'Difference']
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for d, title, ax in zip(data, titles, axes):
    d = d.squeeze()
    vmin, vmax = np.percentile(d, [1, 99])
    ax.imshow(d, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    ax.set_title(title)
plt.tight_layout()
plt.show()

# Show the gradient
grad_siamese = grad_siamese.detach().cpu().numpy().squeeze()[pmln:-pmln, pmln:-pmln]
grad_classic = grad_classic.detach().cpu().numpy().squeeze()[pmln:-pmln, pmln:-pmln]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for d, title, ax in zip([grad_siamese, grad_classic], ['Siamese', 'Classic'], axes):
    vmin, vmax = np.percentile(d, [1, 99])
    ax.imshow(d, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    ax.set_title(title)
plt.tight_layout()
plt.show()