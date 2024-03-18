import os, pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../..")
from seistorch.show import SeisShow
from par2ani import get_thomsen_parameters

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
nz, nx = 256, 256
vel = np.ones((nz, nx), dtype=dtype)*1500
vs = np.ones((nz, nx), dtype=dtype)*1500/1.732
rho = np.ones((nz, nx), dtype=dtype)*1000
Q = np.ones((nz, nx), dtype=dtype)*100

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.
 
src_x = np.linspace(128, 129, 1)
src_z = np.ones_like(src_x)*128

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

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

show.geometry(vel, sources, receivers, savepath="model_geometry.gif", dh=10, interval=1)

# Save the velocity model
vel_path = r"./velocity_model"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "vp.npy"), vel)
np.save(os.path.join(vel_path, "vs.npy"), vs)
np.save(os.path.join(vel_path, "rho.npy"), rho)
np.save(os.path.join(vel_path, "Q.npy"), Q)

c11, c13, c33, c15, c35, c55 = get_thomsen_parameters(vel, vs, rho, 0.1, 0, 0.08, _theta=45.)
np.save(os.path.join(vel_path, "c11.npy"),c11)
np.save(os.path.join(vel_path, "c13.npy"),c13)
np.save(os.path.join(vel_path, "c33.npy"),c33)
np.save(os.path.join(vel_path, "c15.npy"),c15)
np.save(os.path.join(vel_path, "c35.npy"),c35)
np.save(os.path.join(vel_path, "c55.npy"),c55)


# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
