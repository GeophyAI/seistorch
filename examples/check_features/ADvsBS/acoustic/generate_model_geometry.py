import os, pickle
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

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
nz, nx = 64, 128
true = np.ones((nz, nx), dtype=dtype)*1500
true[32:, :] = 2000

# Smooth the velocity model
init = true.copy()
for i in range(100):
    gaussian_filter(init, sigma=1, output=init)

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.
 
src_x = np.linspace(4, nx-4, 10)
src_z = np.ones_like(src_x)

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, nx, 4)
receiver_locz = np.ones_like(receiver_locx)*5

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."

fig, axes= plt.subplots(1, 3, figsize=(8, 4))
# Plot the velocity model and the source and receiver list
for ax, d in zip(axes.ravel(), [true, init]):
    ax.imshow(d, cmap="seismic", aspect='auto', extent=[0, nx, nz, 0])

    ax.scatter([src[0] for src in sources], [src[1] for src in sources], 
                c="r", marker="v", label="Sources")
    ax.scatter(receivers[0][0], receivers[0][1], s=4, c="b", marker="^", 
                label="Receivers")
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("z (grid)")
    ax.legend()

axes[0].set_title("Ground truth")
axes[1].set_title("Initial")

axes[2].plot(true[:,64], label='True')
axes[2].plot(init[:,64], label='Initial')
axes[2].set_title("Trace data")
axes[2].legend()

plt.tight_layout()
plt.savefig("model_geometry.png", dpi=300)
plt.show()

# Save the velocity model
vel_path = r"./velocity_model"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "true.npy"), true)
np.save(os.path.join(vel_path, "init.npy"), init)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
