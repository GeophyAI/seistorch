import os, pickle
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append("../../..")
from seistorch.show import SeisShow
from scipy.ndimage import gaussian_filter

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

show = SeisShow()

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
nz, nx = 64, 128
vel = np.ones((nz, nx), dtype=dtype)*2000.
vel[0:32,:] = 1500.
rho = np.ones((nz, nx), dtype=dtype)*1500
rho[0:32,:] = 1000.

sm_vel = vel.copy()
sm_rho = rho.copy()
for i in range(100):
    gaussian_filter(sm_vel, sigma=1, output=sm_vel)
    gaussian_filter(sm_rho, sigma=1, output=sm_rho)

plt.plot(vel[:,64], label="Original")
plt.plot(sm_vel[:,64], label="Smoothed")
plt.legend()
plt.show()

z = rho*vel
rx = np.gradient(z, axis=1)/(2*z)
rz = np.gradient(z, axis=0)/(2*z)

fig,axes=plt.subplots(1,2,figsize=(8,4))
axes[0].set_title("rx")
axes[1].set_title("rz")
# add colorbar
plt.colorbar(axes[0].imshow(rx, cmap="seismic", aspect='auto'), ax=axes[0])
plt.colorbar(axes[1].imshow(rz, cmap="seismic", aspect='auto'), ax=axes[1])
plt.show()
    
# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.
 
src_x = np.linspace(64, 65, 1)
src_z = np.ones_like(src_x)*16

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, nx, 4)
receiver_locz = np.ones_like(receiver_locx)*1

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."

show.geometry(vel, sources, receivers, savepath="model_geometry.gif", dh=10, interval=1)

# Save the velocity model
vel_path = r"./velocity_model"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "vp.npy"), vel)
np.save(os.path.join(vel_path, "rho.npy"), rho)
np.save(os.path.join(vel_path, "vp_smooth.npy"), sm_vel)
np.save(os.path.join(vel_path, "rho_smooth.npy"), sm_rho)
np.save(os.path.join(vel_path, "rx.npy"), rx)
np.save(os.path.join(vel_path, "rz.npy"), rz)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
