import numpy as np
import matplotlib.pyplot as plt
import h5py
def read_hdf5(path, shot_no):
    with h5py.File(path, 'r') as f:
        return f[f'shot_{shot_no}'][:]

acoustic = read_hdf5('shot_gather_acoustic_fwim.hdf5', 0)
acoustic_rho = read_hdf5('shot_gather_acoustic_rho.hdf5', 0)

fig, axes=plt.subplots(1,2,figsize=(8,4))
axes[0].imshow(acoustic, cmap="seismic", aspect='auto')
axes[0].set_title("Acoustic")
axes[1].imshow(acoustic_rho, cmap="seismic", aspect='auto')
axes[1].set_title("Acoustic variable density")
plt.show()

mid_trace = acoustic.shape[1]//2
plt.plot(acoustic[:,mid_trace], 'r', label='acoustic')
plt.plot(acoustic_rho[:,mid_trace], 'b', label='acoustic variable density')
plt.xlabel('Time samples')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

error = acoustic[:,mid_trace]-acoustic_rho[:,mid_trace]
plt.plot(error, 'black', label='Error')
plt.xlabel('Time samples')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
