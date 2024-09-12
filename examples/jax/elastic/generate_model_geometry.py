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
nz, nx = 64, 129
true_vp = np.ones((nz, nx), dtype=dtype)*1500
true_vp[32:, :] = 2000

true_vs = true_vp/1.73
rho = np.ones_like(true_vp)*2000

# Smooth the velocity model
init_vp = true_vp.copy()
for i in range(400):
    gaussian_filter(init_vp, sigma=1, output=init_vp)

init_vs = init_vp/1.73
# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.
 
src_x = np.linspace(64, 65, 1)
src_z = np.ones_like(src_x)
print(src_x, src_z)
sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, 129, 1)
receiver_locz = np.ones_like(receiver_locx)*5

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."
vmin, vmax=true_vp.min(), true_vp.max()
fig, axes= plt.subplots(2, 3, figsize=(10, 6))
titles = ["True vp", "True vs", "True rho", "Initial vp", "Initial vs", "Initial rho"]
# Plot the velocity model and the source and receiver list
for ax, d, title in zip(axes.ravel(), [true_vp, true_vs, rho, init_vp, init_vs, rho], titles):
    _ax_=ax.imshow(d, cmap="seismic", aspect='auto', extent=[0, nx, nz, 0])
    ax.scatter([src[0] for src in sources], [src[1] for src in sources], 
                c="r", marker="v", label="Sources")
    ax.scatter(receivers[0][0], receivers[0][1], s=4, c="b", marker="^", 
                label="Receivers")
    ax.set_title(title)
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("z (grid)")
    ax.legend()
    plt.colorbar(_ax_, ax=ax)

plt.tight_layout()
plt.savefig("model_geometry.png", dpi=300)
plt.show()

# Save the velocity model
vel_path = r"./velocity_model"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "true_vp.npy"), true_vp)
np.save(os.path.join(vel_path, "true_vs.npy"), true_vs)

np.save(os.path.join(vel_path, "init_vp.npy"), init_vp)
np.save(os.path.join(vel_path, "init_vs.npy"), init_vs)

np.save(os.path.join(vel_path, "rho.npy"), rho)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)