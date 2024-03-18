import os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../../..")
from seistorch.show import SeisShow
from scipy.ndimage import gaussian_filter1d

show = SeisShow()

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
nx, ny, nz = 401, 401, 128
vel = np.ones((nx, nz, ny), dtype=dtype)*1500
vel[:, 64:, :] = 2000

velslice = vel[:, 0, :]

# Smooth the velocity model
velsmooth = vel.copy()
for i in range(5):
    gaussian_filter1d(velsmooth, sigma=10, axis=1, output=velsmooth)

fig, ax = plt.subplots(1,1)
ax.plot(vel[200, :,200], label="True")
ax.plot(velsmooth[200, :,200], label="Smoothed")
ax.legend()
plt.show()

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

sx = np.arange(200, 201, 1)
sy = np.arange(50, ny-50, 80)

srcx_grid, srcy_grid = np.meshgrid(sx, sy)
source_z = 1
#sources = [[x,y,source_z] for x,y in zip(srcx_grid.flatten().tolist(), srcy_grid.flatten().tolist())]
sources = []

# Receivers: [[0, 1, ..., 255], [5, 5, ..., 5], 
#            [0, 1, ..., 255], [5, 5, ..., 5],    
#            [0, 1, ..., 255], [5, 5, ..., 5],
#            ],
# cross-line

receiver_depth = 1
rec_num = 240
receivers = []

for ix in range(srcx_grid.shape[0]):

    if ix % 2 ==0:

        for iy in range(srcx_grid.shape[1]):
            srcx, srcy, srcz = srcx_grid[ix, iy], srcy_grid[ix, iy], source_z
            # from np.int32 to python int
            sources.append([srcx.item(), srcy.item(), srcz])

            recx = np.linspace(srcx-1, srcx-rec_num, rec_num)
            recy = np.ones(rec_num)*srcy
            recz = np.ones_like(recy)*receiver_depth

            receivers.append([recx.flatten().tolist(), recy.flatten().tolist(), recz.flatten().tolist()])

    else:

        for iy in range(srcx_grid.shape[1]-1, -1, -1):
            srcx, srcy, srcz = srcx_grid[ix, iy], srcy_grid[ix, iy], source_z
            sources.append([srcx.item(), srcy.item(), srcz])

            recx = np.linspace(srcx, srcx+rec_num-1, rec_num)
            recy = np.ones(rec_num)*srcy
            recz = np.ones_like(recy)*receiver_depth

            receivers.append([recx.flatten().tolist(), recy.flatten().tolist(), recz.flatten().tolist()])

print(f"Total {len(sources)} sources")

# Show the geometry
show.geometry(velslice, sources, receivers, "./geometry.gif", dh=12.5, interval=100)

# Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)

model_savepath = r"./velocity_model"
os.makedirs(model_savepath, exist_ok=True)
np.save(f"{model_savepath}/true.npy", vel)
np.save(f"{model_savepath}/smooth.npy", velsmooth)