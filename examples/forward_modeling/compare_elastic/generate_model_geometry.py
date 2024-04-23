import os, pickle
import numpy as np
import matplotlib.pyplot as plt

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Generate a two layer velocity model (vp)
# |-----------------------------|
# |                             |
# |vp:1500   vs:867   rho:2000  |
# |                             |
# |-----------------------------|
# |                             |
# |vp:2000   vs:1156  rho:2000  |
# |                             |
# |-----------------------------|
dtype = np.float32
nz, nx = 400, 600
layer = int (nz//2)
vp = np.ones((nz, nx), dtype=dtype)*1500.
vp[layer:, :] = 2500.

vs = np.ones((nz, nx), dtype=dtype)*2500/1.73
vs[0:layer+0, :] = 0.

rho = np.ones_like(vp)*2100
rho[0:layer+0, :] = 1000.

vp_bg = np.ones_like(vp)*1500
vs_bg = vp_bg/1.5

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.
 
src_x = np.linspace(int(nx//2),int(nx//2)-1, 1)
src_z = np.ones_like(src_x)*100

sources = [[src_x, src_z] for src_x, src_z in zip(src_x.tolist(), src_z.tolist())]

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
receiver_locx = np.arange(0, nx, 4)
receiver_locz = np.ones_like(receiver_locx)*100

# The receivers are fixed at the bottom of the model (z=5)
receivers = [[receiver_locx.tolist(), receiver_locz.tolist()]]*len(sources)

assert len(sources) == len(receivers), \
        "The number of sources and receivers must be the same."
titles=["Vp (m/s)", "Vs (m/s)", "Density (kg/m^3)"]
# Plot the velocity model and the source and receiver list
fig, axes=plt.subplots(1,3,figsize=(12,4))
for d, ax, title in zip([vp, vs, rho], axes.ravel(), titles):
    _ax_ = ax.imshow(d, cmap="seismic", aspect='auto', extent=[0, nx, nz, 0])
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("z (grid)")
    ax.scatter([src[0] for src in sources], [src[1] for src in sources], 
                c="r", marker="v", label="Sources")
    ax.scatter(receivers[0][0], receivers[0][1], s=4, c="b", marker="^", 
                label="Receivers")
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("z (grid)")
    plt.colorbar(_ax_, ax=ax)
    ax.legend()
    ax.set_title(title)
plt.tight_layout()
plt.savefig("model_geometry.png", dpi=300)
plt.show()

# Save the velocity model
vel_path = r"./models"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "vp.npy"), vp)
np.save(os.path.join(vel_path, "vs.npy"), vs)
np.save(os.path.join(vel_path, "rho.npy"), rho)

np.save(os.path.join(vel_path, "vp_bg.npy"), vp_bg)
np.save(os.path.join(vel_path, "vs_bg.npy"), vs_bg)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
