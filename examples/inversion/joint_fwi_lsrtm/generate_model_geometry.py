import os, pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)
os.makedirs('figures', exist_ok=True)
# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.

vel = np.load("../../models/marmousi_model/true_vp.npy")
# The depth of the sea is 24*20=480m
rho = 0.31*vel**0.25*1000.

os.makedirs('models', exist_ok=True)
# Smooth the models
vp_smooth = vel.copy()
rho_smooth = rho.copy()
for i in range(20):
    gaussian_filter(vp_smooth, sigma=3, output=vp_smooth)
    gaussian_filter(rho_smooth, sigma=3, output=rho_smooth)
    vp_smooth[0:24] = vel[:24]
    rho_smooth[0:24] = rho[:24]

dh = 20
z = rho*vel
rx = -z*np.gradient(1/z, axis=1)/(2*dh)
rz = -z*np.gradient(1/z, axis=0)/(2*dh)

z_sm = rho_smooth*vp_smooth
rx_sm = -z_sm*np.gradient(1/z_sm, axis=1)/(2*dh)
rz_sm = -z_sm*np.gradient(1/z_sm, axis=0)/(2*dh)

fig,axes=plt.subplots(1,2,figsize=(8,3))
axes[0].set_title("rx")
axes[1].set_title("rz")
# add colorbar
plt.colorbar(axes[0].imshow(rx, cmap="seismic", aspect='auto'), ax=axes[0])
plt.colorbar(axes[1].imshow(rz, cmap="seismic", aspect='auto'), ax=axes[1])
plt.tight_layout()
plt.savefig("figures/true_rx_rz.png", dpi=300, bbox_inches='tight')
plt.show()

plt.plot(rz[:,100])
plt.show()

fig,axes=plt.subplots(2,1,figsize=(5,8))
axes[0].set_title("rx smooth")
axes[1].set_title("rz smooth")
# add colorbar
plt.colorbar(axes[0].imshow(rx_sm, cmap="seismic", aspect='auto'), ax=axes[0])
plt.colorbar(axes[1].imshow(rz_sm, cmap="seismic", aspect='auto'), ax=axes[1])
plt.show()

plt.plot(rz[:,100], label="Original")
plt.plot(rz_sm[:,100], label="Smoothed")
plt.legend()
plt.show()

zero_m = np.zeros_like(vel)

seabed = np.ones_like(vel)
seabed[:24] = 0

np.save("models/true_vp.npy", vel)
np.save("models/true_rho.npy", rho)
np.save("models/smooth_vp.npy", vp_smooth)
np.save("models/smooth_rho.npy", rho_smooth)
np.save("models/zero_m.npy", zero_m)
np.save("models/seabed.npy", seabed)
np.save("models/rx.npy", rx)
np.save("models/rz.npy", rz)
np.save("models/smooth_rx.npy", rx_sm)
np.save("models/smooth_rz.npy", rz_sm)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(vel, cmap="seismic", aspect='auto')
axes[0].set_title("Original")
axes[1].imshow(vp_smooth, cmap="seismic", aspect='auto')
axes[1].set_title("Smoothed")
plt.savefig("figures/velocity_models.png", dpi=300, bbox_inches='tight')
plt.show()

plt.plot(vel[:, 64], label="Original")
plt.plot(vp_smooth[:, 64], label="Smoothed")
plt.legend()
plt.show()

nz, nx =  vel.shape
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
plt.savefig("figures/model_geometry.png", dpi=300)
plt.show()

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
