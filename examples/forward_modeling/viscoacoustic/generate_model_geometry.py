import os, pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../..")
from seistorch.show import SeisShow

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
nz, nx = 128, 128
vel = np.ones((nz, nx), dtype=dtype)*1500
# vel[64:, :] = 2000

rho = np.ones((nz, nx), dtype=dtype)*1000
# rho[64:, :] = 1000

Q = np.ones((nz, nx), dtype=dtype)*100
# Q[64:, :] = 200
# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.

src_x = np.linspace(32, 32, 1)
src_z = np.ones_like(src_x)*32

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, 128, 1)
receiver_locz = np.ones_like(receiver_locx)*5

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
np.save(os.path.join(vel_path, "Q.npy"), Q)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
