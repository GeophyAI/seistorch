import h5py
import matplotlib.pyplot as plt
import numpy as np

def read_hdf5(path, shot_no):
    with h5py.File(path, 'r') as f:
        d = f[f'shot_{shot_no}'][...].copy()
    return d

data = read_hdf5('observed.hdf5', 0)
print(data.shape)
vmin,vmax=np.percentile(data, [2, 98])
plt.imshow(data[...,1], cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
plt.show()