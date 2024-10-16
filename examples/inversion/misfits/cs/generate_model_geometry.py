import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
sys.path.append("../../../..")
from seistorch.show import SeisShow
from seistorch.utils import ricker_wave

show = SeisShow()

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Generate the source and receiver list
# Please note that in Seistorch, 
# the coordinates of source points and receiver points are 
# specified in a grid coordinate system, not in real-world distance coordinates. 
# This distinction is essential for accurate simulation and interpretation of results.

vel = np.load("../../../models/marmousi_model/true_vp.npy")
# The depth of the sea is 24*20=480m

nz, nx =  vel.shape
expand = 50

# The model is expanded by 50 grid points 
# in left and right direction for better illuminating.

num_receivers = 128 # the number of receivers
rec_depth = 1 # the depth of receivers
sources, receivers = [], []
srcx_recx_offset = 5 # the offset between the source and the first receiver
current_srcx = 0 
src_x_inverval = 5 # the grid interval of source
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
dh=20

show.geometry(vel, sources, receivers, savepath="geometry.gif", dh=dh, interval=1)

# # Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)

wavelet_forward = ricker_wave(fm=10, dt=0.001, T=4000, delay=256, dtype="numpy")*5
wavelet_inversion = ricker_wave(fm=10, dt=0.001, T=4000, delay=256, dtype="numpy")
fig,ax = plt.subplots(figsize=(6,3))
ax.plot(wavelet_forward, label="wavelet for forward")
ax.plot(wavelet_inversion, label="wavelet for inversion")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig("wavelet.png",dpi=300)
# Save the wavelet
save_path = r"./wavelet"
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, "wavelet_forward.npy"), wavelet_forward)
np.save(os.path.join(save_path, "wavelet_inversion.npy"), wavelet_inversion)