import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.

vp = np.load("../../models/marmousi_model/true_vp.npy")

# vp = np.ones((256, 512), dtype=np.float32)*1500.
# vp[128:] = 2500.

vs = vp/1.73
zero_ref = np.zeros_like(vp)
init_vp = vp.copy()
init_vs = vs.copy()
seabed = np.ones_like(vp)
seabed[0:20] = 0.

for i in range(1):
    init_vp = gaussian_filter(init_vp, sigma=5)
    init_vs = gaussian_filter(init_vs, sigma=5)
    init_vp[0:24] = 1500.
    init_vs[0:24] = 1500./1.73

true_mvp = 2*(vp-init_vp)/init_vp
true_mvs = 2*(vs-init_vs)/init_vs

rho = np.ones_like(vp)*1000.0

os.makedirs("velocity", exist_ok=True)
np.save("velocity/true_vp.npy", vp)
np.save("velocity/true_vs.npy", vs)
np.save("velocity/rho.npy", rho)
np.save("velocity/true_mvp.npy", true_mvp)
np.save("velocity/true_mvs.npy", true_mvs)
np.save("velocity/smooth_vp.npy", init_vp)
np.save("velocity/smooth_vs.npy", init_vs)
np.save("velocity/zero_ref.npy", zero_ref)
np.save("velocity/seabed.npy", seabed)

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
receiver_locz = np.ones_like(receiver_locx)*1

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."

# Plot the velocity model and the source and receiver list
plt.imshow(init_vp, cmap="seismic", aspect='auto', extent=[0, nx, nz, 0])

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
