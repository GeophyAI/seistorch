import torch, tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from configure import *
from implicit_loss import ImplicitLoss

torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
# Load velocity
vel = np.load("../../models/marmousi_model/linear_vp.npy")
true = np.load("../../models/marmousi_model/true_vp.npy")
true = true[:, expand:-expand][::model_scale, ::model_scale]
vel = vel[:, expand:-expand][::model_scale, ::model_scale]
vel = np.pad(vel, ((pmln, pmln), (pmln, pmln)), mode="edge")
pmlc = generate_pml_coefficients_2d(vel.shape, N=pmln, multiple=False)
vel = torch.from_numpy(vel).float().to(dev)
vel.requires_grad = True
domain = vel.shape
nz, nx = domain
# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
# Loss func
implicit_loss = ImplicitLoss()
# Optimizer
opt = torch.optim.Adam([{'params': [vel], 'lr': lr_vel}])
# Geometry
src_x = np.arange(pmln, nx-pmln, srcx_step)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]
# load observed data
obs = np.load("obs_filtered.npy")
obs = torch.from_numpy(obs).float().to(dev)
LOSS = []
MERROR = []

kwargs = dict(wave=wave, b=pmlc, src_list = np.array(sources), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, pmln=pmln)
kwargs_imshow = dict(vmin=vmin, vmax=vmax, aspect='auto', cmap='seismic', extent=[0, (nx-2*pmln)*dh, (nz-2*pmln)*dh, 0])
for epoch in tqdm.trange(EPOCHS):
    # Select part of the data
    rand_shots = np.random.randint(0, len(sources), size=batch_size).tolist()
    kwargs.update(dict(src_list=np.array(sources)[rand_shots]))
    syn = forward(c=vel, **kwargs)
    # Loss
    _obs = obs[rand_shots].unsqueeze(3)
    _syn = syn.unsqueeze(3)
    loss = implicit_loss(_obs, _syn)
    opt.zero_grad()
    loss.backward()
    if reset_water:
        vel.grad[:water_grid] = 0.
    opt.step()
    LOSS.append(loss.item())
    inverted = vel.cpu().detach().numpy()[pmln:-pmln, pmln:-pmln]
    MERROR.append(np.sum((true - inverted)**2))
    if epoch % show_every == 0: 
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        plt.imshow(inverted, **kwargs_imshow)
        plt.show()

plt.plot(LOSS)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.plot(MERROR)
plt.xlabel("Epoch")
plt.ylabel("Model error")
plt.show()
np.save("model_error_by_implicitloss.npy", np.array(MERROR))
np.save("inverted_by_vggloss.npy", vel.cpu().detach().numpy()[pmln:-pmln, pmln:-pmln])

