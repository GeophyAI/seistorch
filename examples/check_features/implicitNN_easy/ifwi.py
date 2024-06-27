import torch, os, tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils_torch import imshow, forward, ricker, showgeom, show_gathers
from utils_torch import write_pkl
from siren import Siren

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
# configure
model_scale = 2 # 1/2
expand = 50
expand = int(expand/model_scale)
delay = 150 # ms
fm = 5 # Hz
dt = 0.002 # s
nt = 1500 # timesteps
dh = 20 # m
srcz = 1 # grid point
recz = 1 # grid point
# Configures for implicit neural network
in_features = 2
out_features = 1
hidden_features = 128
hidden_layers = 6
vmin = 1000.
vmax = 5500.
l2loss = torch.nn.MSELoss()
lr = 5e-5
EPOCHS = 50000
show_every = 1000

# Load velocity
vel = np.load("../../models/marmousi_model/true_vp.npy")
vel = vel[:, expand:-expand][::model_scale, ::model_scale]
domain = vel.shape
nz, nx = domain

# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
# show spectrum
plt.figure(figsize=(5, 3))
amp = np.abs(np.fft.fft(wave.cpu().numpy()))
freqs = np.fft.fftfreq(nt, dt)
amp = amp[freqs >= 0]
freqs = freqs[freqs >= 0]
plt.plot(freqs[:50], amp[:50])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Geometry
src_x = np.linspace(0, nx-1, 20)
src_z = np.ones_like(src_x)*srcz

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, nx, 1)
receiver_locz = np.ones_like(receiver_locx)*recz

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)

# show geometry
showgeom(vel, sources, receivers, figsize=(5, 4))
print(f"The number of sources: {len(sources)}")
print(f"The number of receivers: {len(receivers[0])}")

# forward
#  for observed data
# To GPU
vel = torch.from_numpy(vel).float().to(dev)
kwargs = dict(wave=wave, src_list = np.array(sources), domain=domain, dt=dt, h=dh, dev=dev, recz=recz, savewavefield=False)
with torch.no_grad():
    rec_obs = forward(c=vel,**kwargs)
# Show gathers
show_gathers(rec_obs.cpu().numpy(), size=1, figsize=(5, 5))

# implicit neural network for representing velocity
imvel = Siren(in_features=in_features, 
              out_features=out_features, 
              hidden_features=hidden_features, 
              hidden_layers=hidden_layers, 
              outermost_linear=True,
              domain_shape=vel.shape).to(dev)
opt = torch.optim.Adam(imvel.parameters(), lr=lr)
coords = imvel.coords.to(dev)
vp = imvel(coords)[0]
# Denormalize
std, mean = 1000., 3000.
# forward for predicted data
LOSS = []
for epoch in tqdm.trange(EPOCHS):
    vp = imvel(coords)[0]
    # Denormalize
    vp = vp * std + mean
    vp[0:12] = 1500.
    # forward
    rec_pred = forward(c=vp, **kwargs)
    loss = l2loss(rec_pred, rec_obs)
    LOSS.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % show_every == 0:
        plt.figure(2, figsize=(6, 3))
        ax = plt.subplot(211)
        ax.imshow(vp.cpu().detach().numpy().reshape(vel.shape), cmap="seismic", aspect="auto")
        # show loss
        ax = plt.subplot(212)
        ax.plot(LOSS)
        plt.yscale("log")
        plt.tight_layout()
        plt.show()



