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

vp = np.load("../../../models/marmousi_model/true_vp.npy")
vs = np.load("../../../models/marmousi_model/true_vs.npy")
rho = np.load("../../../models/marmousi_model/rho.npy")

initvp = np.load("../../../models/marmousi_model/linear_vp.npy")
initvs = np.load("../../../models/marmousi_model/linear_vs.npy")

nz, nx =  vp.shape
expand = 50
# The model is expanded by 50 grid points 
# in left and right direction for better illuminating.
src_x = np.arange(expand, nx-expand, 5)
src_z = np.ones_like(src_x)

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(expand, nx-expand, 1)
receiver_locz = np.ones_like(receiver_locx)*5

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
titles = ["vp", "vs", "rho", "init_vp", "init_vs", "rho"]
for d, ax, title in zip([vp, vs, rho, initvp, initvs, rho], axes.ravel(), titles):
    ax.imshow(d, cmap='seismic', aspect='auto', extent=[0, nx, nz, 0])
    ax.set_title(title)
    ax.scatter([src[0] for src in sources], [src[1] for src in sources], 
                c="r", marker="v", label="Sources")
    ax.scatter(receivers[0][0], receivers[0][1], s=4, c="b", marker="^", 
                label="Receivers")
    ax.legend()
plt.tight_layout()
plt.savefig("model_geometry.png", dpi=300)
plt.show()

print(f"The number of sources is {len(sources)}")

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
