import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

true = np.load("/mnt/others/DATA/Inversion/RNN//velocity/true_vp.npy")

# init = np.zeros_like(true)
init = np.mean(true, axis=1, keepdims =True)
init = gaussian_filter1d(init, sigma=1, axis=0)

init = np.repeat(init, true.shape[1], axis=1)

np.save("/mnt/others/DATA/Inversion/RNN//velocity/linear.npy",
         init)

plt.imshow(true, aspect="auto", cmap=plt.cm.seismic)
plt.show()
plt.imshow(init, aspect="auto", cmap=plt.cm.seismic)
plt.show()

plt.plot(init[:,100])
plt.show()