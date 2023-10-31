import os, pickle
import numpy as np
import matplotlib.pyplot as plt

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.

vel = np.load("../marmousi_model/true_vp.npy")
# The depth of the sea is 24*20=480m
seabed = np.ones_like(vel)
seabed[0:24] = 0
np.save("../marmousi_model/seabed.npy", seabed)
nz, nx =  vel.shape
expand = 50
# The model is expanded by 50 grid points 
# in left and right direction for better illuminating.
src_x = np.linspace(expand, nx-expand, 3)
src_z = np.ones_like(src_x)

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(expand, 250, 50)
receiver_locz = np.ones_like(receiver_locx)*5

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."

# Plot the velocity model and the source and receiver list
plt.imshow(vel, cmap="seismic", aspect='auto', extent=[0, nx, nz, 0])

plt.scatter([src[0] for src in sources], [src[1] for src in sources], 
            c="r", marker="v", label="Sources")
plt.scatter(receivers[0][0], receivers[0][1], s=4, c="b", marker="^", 
            label="Receivers")
plt.legend()
plt.xlabel("x (grid)")
plt.ylabel("z (grid)")
plt.title(f"Model and Geometry\n \
          {len(sources)} sources, {len(receivers[0][0])} receivers)")
plt.savefig("model_geometry.png", dpi=300)
plt.show()

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
