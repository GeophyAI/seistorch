import numpy as np
import segyio, torch
import matplotlib.pyplot as plt

d = np.load("/mnt/others/DATA/Inversion/RNN/data/marmousi_viscoacoustic.npy")
vmin,vmax=np.percentile(d, [5,95])
plt.imshow(d[0].squeeze(), vmin=vmin, vmax=vmax, cmap=plt.cm.gray, aspect="auto")
plt.show()
plt.plot(d[0][:,2,0])
plt.show()
print(d.max())
# plt.plot(d[:,100].squeeze())
# plt.show()

