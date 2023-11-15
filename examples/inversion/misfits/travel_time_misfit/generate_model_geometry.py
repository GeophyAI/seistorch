

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

dh = 20
nz, nx = 151, 801

vel1 = np.ones((nz, nx), np.float32)*1500
vel2 = np.ones((nz, nx), np.float32)*1500
seabed = np.ones_like(vel1)

water_depth = 100 # m
water_grid = int(water_depth/dh)
seabed[0:water_grid] = 0

for depth in range(water_grid, nz):
    # let the velocity increasing with depth
    vel1[depth] = 1500 + (depth-water_grid) * 10.
    vel2[depth] = 1500 + (depth-water_grid) * 15.

fig, axes= plt.subplots(3, 1, figsize=(6, 6))
vmin=1500
vmax=max(vel1.max(), vel2.max())
kwargs = dict(cmap='seismic', vmin=vmin, vmax=vmax, aspect="auto", extent=[0, nx*dh, nz*dh, 0])
axes[0].imshow(vel1, **kwargs)
axes[1].imshow(vel2, **kwargs)
axes[0].set_title("True")
axes[1].set_title("Init")
plt.colorbar(axes[0].images[0], ax=axes[0])
plt.colorbar(axes[1].images[0], ax=axes[1])
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Z (m)")
axes[1].set_xlabel("X (m)")
axes[1].set_ylabel("Z (m)")

# plot the velocity difference
depth = np.arange(nz)*dh
axes[2].plot(depth, vel1[:,150], label="True")
axes[2].plot(depth, vel2[:,150], label="Init")
axes[2].set_ylabel("Velocity (m/s)")
axes[2].set_xlabel("Depth (m)")
axes[2].legend()

plt.tight_layout()
plt.show()
fig.savefig("velocity_model.png", bbox_inches="tight", dpi=600)

model_save_path = r"./velocity_model"
os.makedirs(model_save_path, exist_ok=True)
np.save(os.path.join(model_save_path, "true.npy"), vel1)
np.save(os.path.join(model_save_path, "init.npy"), vel2)
np.save(os.path.join(model_save_path, "seabed.npy"), seabed)


num_receivers = 300 # the number of receivers
rec_depth = 1 # the depth of receivers
sources, receivers = [], []
srcx_recx_offset = 5 # the offset between the source and the first receiver
current_srcx = 0 
src_x_inverval = 10 # the grid interval of source
srcz = 1 # the grid depth of source
idx = 0 # the index of source
while current_srcx<nx:
    idx+=1

    srcx = idx*src_x_inverval

    if srcx < num_receivers+srcx_recx_offset:
        continue
    else:
        loc_of_first_recx = srcx - srcx_recx_offset
        recx = np.arange(loc_of_first_recx, loc_of_first_recx-num_receivers, -1)
        recz = np.ones_like(recx)*rec_depth

        sources.append([srcx, srcz])
        receivers.append([recx.tolist(), recz.tolist()])

    current_srcx = srcx

print(f"the number of sources: {len(sources)}")
# Plot the velocity model and the source and receiver list
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(vel1, cmap='seismic', aspect="auto", extent=[0, nx*dh, nz*dh, 0])
sc_sources = ax.scatter([], [], c='red', marker="v", label='Sources')
sc_receivers = ax.scatter([], [], c='blue', marker="^", label='Receivers')
ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
plt.tight_layout()
# define the figure
def update(frame):
    
    sc_sources.set_offsets(np.stack(sources[frame], axis=0).T*dh)
    sc_receivers.set_offsets(np.stack(receivers[frame], axis=1)*dh)

    return sc_sources, sc_receivers

ani = FuncAnimation(fig, update, frames=len(sources), interval=1)  # frames 是迭代次数，interval 是每帧之间的时间间隔
ani.save('geometry.gif', writer='imagemagick')  # 使用 'imagemagick' 作为写入器保存为 GIF 格式

# # Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
