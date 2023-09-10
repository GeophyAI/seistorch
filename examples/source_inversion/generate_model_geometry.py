import os, pickle
import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader

# Using Bspline wavelet as true wavelet
def Bspline(t, fb, m, p, q):
    numerator = np.sqrt(fb) * (np.sinc((fb * t / m)) ** m)
    denominator = q - p
    sinc_term = q * np.sinc(2 * q * t) - p * np.sinc(2 * p * t)
    result = (numerator / denominator) * sinc_term
    return result

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Generate a two layer velocity model (vp)
# |-----------------------------|
# |                             |
# |           1500 m/s          |
# |                             |
# |-----------------------------|
# |                             |
# |           2000 m/s          |
# |                             |
# |-----------------------------|
dtype = np.float32
nz, nx = 128, 256
vel = np.ones((nz, nx), dtype=dtype)*1500
vel[64:, :] = 2000

background_vel = np.ones_like(vel)*1500

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.

sources = [[128, 1]]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, 256, 4)
receiver_locz = np.ones_like(receiver_locx)*5

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."

# True wavelet 
config_path = r"./forward.yml"
# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)

dt = cfg['geom']['dt']
nt = cfg['geom']['nt']
t = np.arange(0, dt*nt, dt)-cfg['geom']['wavelet_delay']*dt
true = Bspline(t=t, fb=10.0, m=5., p=7., q=15)

np.save("wavelet_bspline.npy", true)
fig,axes=plt.subplots(1,3,figsize=(10,5))
# Plot the velocity model and the source and receiver list
axes[0].imshow(vel, cmap="seismic", aspect='auto', extent=[0, nx, nz, 0])
axes[1].imshow(background_vel, cmap="seismic", aspect='auto', extent=[0, nx, nz, 0])
for ax in axes.ravel()[:2]:
    ax.scatter([src[0] for src in sources], [src[1] for src in sources], 
            c="r", marker="v", label="Sources")
    ax.scatter(receivers[0][0], receivers[0][1], s=4, c="b", marker="^", 
            label="Receivers")
    ax.legend()
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("z (grid)")
    ax.set_title("Model and Geometry")
axes[2].plot(true, label="True wavelet")
axes[2].legend()
plt.tight_layout()
plt.savefig("model_geometry.png", dpi=300)
plt.show()

# Save the velocity model
vel_path = r"./velocity_model"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "true_vp.npy"), vel)
np.save(os.path.join(vel_path, "back_vp.npy"), background_vel)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)



