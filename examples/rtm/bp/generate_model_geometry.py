import os, pickle
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
from seistorch.show import SeisShow
from seistorch.utils import ricker_wave
from seistorch.signal import SeisSignal
from seistorch.io import SeisIO

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

vel = np.load("../models/bp2.5d1997/model.npy")

smvel = np.load("../models/bp2.5d1997/vsmooth.npy")

nz, nx =  vel.shape

# The model is expanded by 50 grid points 
# in left and right direction for better illuminating.

num_receivers = 256 # the number of receivers
rec_depth = 1 # the depth of receivers=1*12.5=12.5m
rec_interval = 2 # the interval of receivers=2*12.5=25m
sources, receivers = [], []
srcx_recx_offset = 0 # the offset between the source and the first receiver
current_srcx = 0 
src_x_inverval = 4 # the grid interval of source
srcz = 1 # the grid depth of source
idx = 0 # the index of source
while current_srcx<nx:

    idx+=1
    srcx = idx*src_x_inverval

    loc_of_first_recx = srcx + srcx_recx_offset
    recx = np.array([loc_of_first_recx+i*rec_interval for i in range(num_receivers)])
    recz = np.ones_like(recx)*rec_depth

    if recx.min()<0 or recx.max()>nx-1:
        break

    sources.append([srcx, srcz])
    receivers.append([recx.tolist(), recz.tolist()])

    current_srcx = srcx



print(f"the number of sources: {len(sources)}")
dh=12.5

show.geometry(smvel, sources, receivers, savepath="geometry.gif", dh=dh, interval=1)

# # Save the source and receiver list
save_path = r"./geometry"
os.makedirs(save_path, exist_ok=True)
write_pkl(os.path.join(save_path, "sources.pkl"), sources)
write_pkl(os.path.join(save_path, "receivers.pkl"), receivers)
