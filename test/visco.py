import numpy as np
import torch
import matplotlib.pyplot as plt

d = np.load("/mnt/data/wangsw/inversion/marmousi/data/marmousi_elastic.npy")
print(d.shape)
# d = np.load("/mnt/others/DATA/Inversion/RNN/coding_visco/ypred.npy")
print(d.shape)
vmin,vmax=np.percentile(d[-1][:,:,-1], [2,98])
plt.imshow(d[-1][:,:,-1].squeeze(), vmin=vmin, vmax=vmax, cmap=plt.cm.gray, aspect="auto")
plt.show()
# plt.plot(d[0][:,2,0])
# plt.show()
print(d.max())
# plt.plot(d[:,100].squeeze())
# plt.show()

