import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import cumulative_trapezoid
import lesio

d = np.load("/mnt/data/wangsw/inversion/marmousi_10m/data/marmousi_elastic_obn.npy")[55][..., 1].squeeze()


d = lesio.tools.fitler_fft(d.copy(), dt=0.001, N=3, low=5, axis=0, mode="lowpass")

counts = 2
d2 = torch.from_numpy(d.copy())
for _ in range(counts):
    d2 = torch.cumulative_trapezoid(d2, axis=0)
d2 = d2.numpy()
print(d.shape, d2.shape)
fig, axes=plt.subplots(1,2,figsize=(10,5))
vmin,vmax=np.percentile(d, [2, 98])
axes[0].imshow(d, vmin=vmin, vmax=vmax, aspect="auto", cmap=plt.cm.seismic)
vmin,vmax=np.percentile(d2, [2, 98])
axes[1].imshow(d2, vmin=vmin, vmax=vmax, aspect="auto", cmap=plt.cm.seismic)

plt.show()

spec, freqs = lesio.tools.freq_spectrum(d, dt=0.001)
spec_integ, freqs_integ = lesio.tools.freq_spectrum(d2, dt=0.001)

plt.plot(freqs[:200], spec[:200]/spec.max())
plt.plot(freqs_integ[:200], spec_integ[:200]/spec_integ.max())
plt.legend(["Original", "Intergral"])
plt.show()