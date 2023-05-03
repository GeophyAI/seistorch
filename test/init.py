import numpy as np
import segyio
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, gaussian_filter1d
# filepath = r"/mnt/others/DATA/Inversion/Models/Marmousi/vp_marmousi-ii.segy"
# with segyio.open(filepath, ignore_geometry=True) as f:
#     f.mmap()
#     vel = []
#     for trace in f.trace:
#         vel.append(trace.copy())


# vel=np.array(vel).T
# print(vel.shape)
# vel = gaussian_filter(vel, sigma=2)

# vel = vel[::25,::25]

# print(vel.shape)
# plt.imshow(vel, aspect="auto", cmap=plt.cm.seismic)
# plt.show()

true = np.load("/mnt/others/DATA/Inversion/RNN/velocity/true_vp.npy")

rho = np.ones_like(true)*2000
q = np.ones_like(true)*100

np.save("/mnt/others/DATA/Inversion/RNN/velocity/rho_const.npy", rho)
np.save("/mnt/others/DATA/Inversion/RNN/velocity/q_const.npy", q)

# init = np.zeros_like(true)
# init = np.mean(true, axis=1, keepdims =True)
# init = gaussian_filter1d(init, sigma=1, axis=0)

# init = np.repeat(init, true.shape[1], axis=1)

# np.save("/mnt/others/DATA/Inversion/RNN//velocity/linear.npy",
#          init)

# plt.imshow(true, aspect="auto", cmap=plt.cm.seismic)
# plt.show()
# plt.imshow(init, aspect="auto", cmap=plt.cm.seismic)
# plt.show()
