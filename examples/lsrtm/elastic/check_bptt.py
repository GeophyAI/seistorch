import numpy as np
import matplotlib.pyplot as plt
import glob
rootpath = r'./wf_pml/'

files = glob.glob(rootpath + '/*.npy')
forward_files = sorted([f for f in files if 'foward' in f])
backward_files = sorted([f for f in files if 'backward' in f])
width = 50
nt = len(forward_files)
t = 2000
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
forward = np.load(forward_files[t])[0][width:-width, width:-width]
backward = np.load(backward_files[nt-t-2])[0][width:-width, width:-width]

assert forward.shape==backward.shape
vmin, vmax=np.percentile(backward, [2, 98])
axes[0].imshow(forward, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
axes[1].imshow(backward, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
axes[2].imshow(forward-backward, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.show()
