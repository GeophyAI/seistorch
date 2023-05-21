import lesio, tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("qt5agg")

d = np.load("/mnt/data/wangsw/inversion/marmousi_10m/data/marmousi_acoustic_obn_constRHO.npy")[55,...,0]

# fd = np.zeros_like(d)

# for i in tqdm.trange(fd.shape[0]):
#     fd[i][:] = lesio.tools.fitler_fft(d[i].copy(), dt=0.001, 
#                                         N=5, low=5, axis=0, mode="highpass")
# print(fd.shape, fd.dtype)
# np.save("/mnt/others/DATA/Inversion/RNN/data/marmousi_acoustic_no5hz.npy", fd)
fd = lesio.tools.fitler_fft(d.copy(), dt=0.001, 
                            N=5, low=5, axis=0, mode="lowpass")
amp_before, freq = lesio.freq_spectrum(d, dt=0.001)
amp_after, freq = lesio.freq_spectrum(fd, dt=0.001)
plt.plot(freq[0:200], amp_before[0:200])
plt.plot(freq[0:200], amp_after[0:200])
plt.vlines([5], ymin=0, ymax=amp_after.max())
plt.legend(["Befoer", "After"])
plt.show()

fig,axes = plt.subplots(1,2, figsize=(5,3))
vmin, vmax = np.percentile(d, [2, 98])
ax0=axes[0].imshow(d.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
vmin, vmax = np.percentile(fd, [2, 98])
ax1=axes[1].imshow(fd.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
plt.show()