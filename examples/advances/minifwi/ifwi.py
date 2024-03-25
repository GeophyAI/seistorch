import torch, tqdm, os
torch.backends.cudnn.benchmark = True
torch.cuda.cudnn_enabled = True

import numpy as np
import matplotlib.pyplot as plt
from utils_torch import imshow, forward, ricker, showgeom, show_gathers, generate_mesh
from networks import Siren

seed = 20231201
torch.manual_seed(seed)
np.random.seed(seed)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configure
model_scale = 2 # 1/2
expand = 50
expand = int(expand/model_scale)
unit = 1#0.001
delay = 100 # ms
fm = 8 # Hz
dt = 0.0019 # s
nt = 1000 # timesteps
dh = 15*unit # m
srcz = 0 # grid point
recz = 0 # grid point
std = 1000*unit # the standard deviation
mean = 3000*unit # the mean
savepath = r"results"
if not os.path.exists(savepath):
    os.makedirs(savepath, exist_ok=True)
# Training
criterion = torch.nn.MSELoss()
lr = 5e-5
decay = 0.9999
epochs = 10000

# Load velocity
true = np.load("../../models/marmousi_model/true_vp.npy")*unit
init = np.load("../../models/marmousi_model/linear_vp.npy")
true = true[24:][::model_scale,::model_scale]

domain = true.shape
nz, nx = domain
# imshow(vel, vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 4))

kwargs_vel = dict(vmin=1500*unit, vmax=5500*unit, cmap="seismic", figsize=(5, 3))

# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)

# Geometry
srcx = np.arange(expand, nx-expand, 20).tolist()
srcz = (np.ones_like(srcx) * srcz).tolist()
src_loc = list(zip(srcx, srcz))

recx = np.arange(expand, nx-expand, 1).tolist()
recz = (np.ones_like(recx) * recz).tolist()
rec_loc = list(zip(recx, recz))

# show geometry
showgeom(true, src_loc, rec_loc, figsize=(5, 4))
print(f"The number of sources: {len(src_loc)}")
print(f"The number of receivers: {len(rec_loc)}")

# forward for observed data
# To GPU
true = torch.from_numpy(true).float().to(dev)
with torch.no_grad():
    rec_obs = forward(wave, true, np.array(src_loc), domain, dt, dh, dev, recz=0)
# Show gathers
show_gathers(rec_obs.cpu().numpy(), figsize=(8, 5))

# To GPU
nn = Siren(in_features=2, 
           out_features=1, 
           hidden_features=128, 
           hidden_layers=4, 
           outermost_linear=True,
           domain_shape=domain, 
           dh=dh).to(dev)

coords = generate_mesh(domain, dh).to(dev)
# Configures for training
opt = torch.optim.Adam(nn.parameters(), lr=lr)
# lr decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)
history = []
# closure for training
def closure():
    opt.zero_grad()
    # rand_size = 10
    # rand_shots = np.random.randint(0, len(src_loc), size=rand_size).tolist()
    vel, _ = nn(coords)
    vel = vel*std+mean
    #rec_syn = forward(wave, vel, np.array(src_loc)[rand_shots], domain, dt, dh, dev, recz=0)
    rec_syn = forward(wave, vel, src_loc, domain, dt, dh, dev, recz=0)
    loss = criterion(rec_syn, rec_obs)
    loss.backward()
    return loss

# training
for epoch in tqdm.trange(epochs):
    loss = opt.step(closure)
    history.append(loss.item())
    scheduler.step()
    # print(f"Epoch: {epoch}, Loss: {loss}")
    
    # show intermediate results every 100 epochs
    if epoch % 500 == 0:
        # save intermediate results
        with torch.no_grad():
            vel, _ = nn(coords)
            vel = vel.cpu().detach().numpy()
            vel = vel*std+mean
        np.save(f"{savepath}/vel_{epoch:04d}.npy", vel)
        imshow(vel, **kwargs_vel)
        fig,axes=plt.subplots(2, 1, figsize=(5, 6))
        axes[0].plot(history)
        axes[1].plot(true.cpu().detach().numpy()[:,100], label="True")
        axes[1].plot(vel[:,100], label="Predict")
        axes[1].legend()
        plt.show()