import numpy as np
import matplotlib.pyplot as plt

acoustic = np.load('shot_gather_acoustic.npy', allow_pickle=True)
acoustic_rho = np.load('shot_gather_acoustic_rho.npy', allow_pickle=True)

fig, axes=plt.subplots(1,2,figsize=(8,10))
axes[0].imshow(acoustic[0], aspect='auto', cmap='seismic')
axes[1].imshow(acoustic_rho[0], aspect='auto', cmap='seismic')
axes[0].set_title('Acoustic')
axes[1].set_title('Acoustic variable density')
plt.tight_layout()
plt.show()

mid_trace = acoustic[0].shape[1]//2
plt.plot(acoustic[0][:,mid_trace], 'r', label='acoustic')
plt.plot(acoustic_rho[0][:,mid_trace], 'b', label='acoustic variable density')
plt.xlabel('Time samples')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

error = acoustic[0][:,mid_trace]-acoustic_rho[0][:,mid_trace]
plt.plot(error, 'black', label='Error')
plt.xlabel('Time samples')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

print(np.allclose(acoustic[0][:,mid_trace], 
                  acoustic_rho[0][:,mid_trace]))