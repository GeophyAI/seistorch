import os, pickle
import numpy as np
import matplotlib.pyplot as plt

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Generate a two layer 3d velocity model (vp)
#    ________
#  /   .    /|
# /________/ |
# | 1500m/s| |
# |________|/|
# | 2000m/s| |
# |________|/
#
dtype = np.float32
nx, ny, nz = 128, 128, 64
vel = np.ones((nx, nz, ny), dtype=dtype)*1500
vel[:, 32:, :] = 2000

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.
# 
# |X----------------------------|
# Y              v              |
# |              v              |
# |              v              |
# |v v v v v v v * v v v v v v v|
# |              v              |
# |              v              |
# |              v              |
# |-----------------------------|

# * represents the location of shots
# v represents the location of receivers

sources = [[nx//2, ny//2, 1]]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
# cross-line
receiver_depth = 1
step = 4
receiver_locx = np.arange(0, nx, step)
receiver_locy = np.ones_like(receiver_locx)*(ny//2)
receiver_locz = np.ones_like(receiver_locx)*receiver_depth
# in-line
receiver_locx = np.concatenate((receiver_locx, np.ones_like(np.arange(0, nx, step))*(nx//2)))
receiver_locy = np.concatenate((receiver_locy, np.arange(0, ny, step)))
receiver_locz = np.concatenate((receiver_locz, np.ones_like(np.arange(0, nx, step))*receiver_depth))

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locy.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."

# Plot the velocity model and the source and receiver list
plt.imshow(vel[:,0,:], cmap="seismic", aspect='auto', extent=[0, nx, ny, 0])

plt.scatter([src[0] for src in sources], [src[1] for src in sources], 
            c="r", marker="v", label="Sources")
plt.scatter(receivers[0][0], receivers[0][1], s=4, c="b", marker="^", 
            label="Receivers")
plt.legend()
plt.xlabel("x (grid)")
plt.ylabel("y (grid)")
plt.title("xoy pline")
plt.savefig("model_geometry.png", dpi=300)
plt.show()

# Save the velocity model
vel_path = r"./velocity_model"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "vp.npy"), vel)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
