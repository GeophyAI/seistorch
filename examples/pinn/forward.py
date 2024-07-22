import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils_torch import imshow, forward, ricker, showgeom, show_gathers

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
# configure
model_scale = 2 # 1/2
expand = 50
expand = int(expand/model_scale)
delay = 150 # ms
fm = 5 # Hz
dt = 0.001 # s
nt = 2000 # timesteps
dh = 10 # m
srcz = 0 # grid point
recz = 0 # grid point
# Training
criterion = torch.nn.MSELoss()
lr = 10.
epochs = 100

# Load velocity
vel = np.load("./velocity_model/vp.npy")
domain = vel.shape
nz, nx = domain

# load wave
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
# show spectrum
plt.figure(figsize=(5, 3))
plt.plot(np.fft.fftfreq(nt, dt), np.abs(np.fft.fft(wave.cpu().numpy())))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Geometry
src_loc = pickle.load(open("./geometry/sources.pkl", "rb"))
rec_loc = pickle.load(open("./geometry/receivers.pkl", "rb"))

# show geometry
showgeom(vel, src_loc, rec_loc, figsize=(5, 4))
print(f"The number of sources: {len(src_loc)}")
print(f"The number of receivers: {len(rec_loc)}")

# forward for observed data
# To GPU
vel = torch.from_numpy(vel).float().to(dev)
with torch.no_grad():
    rec_obs = forward(wave, 
                      vel, 
                      np.array(src_loc), 
                      domain, 
                      dt, 
                      dh, 
                      dev, 
                      recz=0, 
                      savewavefield=True)
# Show gathers
show_gathers(rec_obs.cpu().numpy(), size=1, figsize=(10, 10))