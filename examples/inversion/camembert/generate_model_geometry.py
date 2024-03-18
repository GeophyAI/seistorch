import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.

# Define the grid size
nx = 201 # in grid
nz = 201 # in grid
dx = dz = 10 #m
num_receivers = 201 # the number of receivers
num_sources = 11 # the number of receivers

inside_velocity = 3600.
outside_velocity = 3000.

# Define the background model
model = np.ones((nz, nx), dtype=np.float32)*outside_velocity
init = model.copy()
# center of the circle
center_x = nx // 2
center_y = nz // 2

radius=int(600//dx)

for y in range(nz):
    for x in range(nx):
        if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
            model[y, x] = inside_velocity

# Save model
# Save the velocity model
vel_path = r"./velocity_model"
os.makedirs(vel_path, exist_ok=True)
np.save(os.path.join(vel_path, "true.npy"), model)
np.save(os.path.join(vel_path, "init.npy"), init)

# Setup the source and receiver list
sources, receivers = [], []

src_x = np.linspace(5,nx-5,num_sources).tolist()
src_z = np.ones_like(src_x).tolist()

sources = [[src_x[i], src_z[i]] for i in range(len(src_x))]
rec_depth = nz-1 # the depth of receivers
rec_x = np.arange(0,nx,1)
rec_z = np.ones_like(rec_x)*rec_depth
receivers = [[rec_x.tolist(), rec_z.tolist()]]*len(sources)


# Plot the velocity model and the source and receiver list
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
extent = [0, nx*dx, nz*dx, 0]
ax0=axes[0].imshow(model, cmap="jet", aspect='auto', extent=extent)
ax1=axes[1].imshow(init, vmin=model.min(), vmax=model.max(), cmap="jet", aspect='auto', extent=extent)

for ax in axes.ravel():

    ax.scatter([src[0]*dx for src in sources], [src[1]*dx for src in sources], 
                c="r", marker="v", label="Sources")
    ax.scatter(np.array(receivers[0][0])*dx, np.array(receivers[0][1])*dx, s=4, c="green", marker="^", 
                label="Receivers")
    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")


axes[0].set_title(f"True model")
axes[1].set_title(f"Background model")

print(f"Total number of sources: {len(sources)}")
print(f"Total number of receivers: {len(receivers[0][0])}")

plt.colorbar(ax0, ax=axes[0], shrink=0.8)
plt.colorbar(ax1, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.savefig("model_geometry.png", dpi=300, bbox_inches="tight")
plt.show()

# # Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
