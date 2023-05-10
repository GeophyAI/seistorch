import numpy as np
# import segyio
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

true = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vp.npy")
nz, nx = true.shape
print(true.min(), true.max())
linear = np.copy(true)
up = (4400-1500)/(nz-24)
for i in range(nz-24):
    linear[24+i] = 1500+i*up
ax=plt.imshow(linear)
np.save("/mnt/data/wangsw/inversion/marmousi_20m/velocity/linear_vp.npy", linear)
linear_vs = linear/1.73
linear_vs[0:24,:] = 0
print(linear_vs.max(), linear_vs.min())
np.save("/mnt/data/wangsw/inversion/marmousi_20m/velocity/linear_vs.npy", linear_vs)

# rho = np.ones_like(true)*2000
# q = np.ones_like(true)*100

# np.save("/mnt/others/DATA/Inversion/RNN/velocity/rho_const.npy", rho)
# np.save("/mnt/others/DATA/Inversion/RNN/velocity/q_const.npy", q)

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
