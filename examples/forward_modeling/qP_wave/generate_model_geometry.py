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
nz, nx = 301, 301
vel = np.ones((nz, nx), dtype=dtype)*3000.
# vs = np.ones((nz, nx), dtype=dtype)*3000./1.73
# rho = np.ones((nz, nx), dtype=dtype)*1000
# Q = np.ones((nz, nx), dtype=dtype)*20

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.
 
src_x = np.linspace(150, 151, 1)
src_z = np.ones_like(src_x)*150

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, 301, 4)
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

# Model A
epsilons_a = np.ones_like(vel)*0.3
deltas_a = np.ones_like(vel)*0.3

np.save(os.path.join(vel_path, "epsilon_a.npy"), epsilons_a)
np.save(os.path.join(vel_path, "delta_a.npy"), deltas_a)

# Model B
epsilons_b = np.ones_like(vel)*0.3
deltas_b = np.ones_like(vel)*0.1

np.save(os.path.join(vel_path, "epsilon_b.npy"), epsilons_b)
np.save(os.path.join(vel_path, "delta_b.npy"), deltas_b)

# Model C
epsilons_c = np.ones_like(vel)*0.1
deltas_c = np.ones_like(vel)*0.3

np.save(os.path.join(vel_path, "epsilon_c.npy"), epsilons_c)
np.save(os.path.join(vel_path, "delta_c.npy"), deltas_c)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
