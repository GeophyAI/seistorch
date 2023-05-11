import numpy as np
import matplotlib.pyplot as plt

d = np.load("/public1/home/wangsw/FWI/EFWI/Marmousi/marmousi1_20m/data/marmousi_obn.npy")
nn = np.random.normal(0, 0.05, d.shape)
nd = d+nn
no=50
fig,axes=plt.subplots(1,2)
vmin,vmax=np.percentile(d[no], [2, 98])
axes[0].imshow(d[no][...,0],vmin=vmin,vmax=vmax,cmap=plt.cm.seismic, aspect="auto")
axes[1].imshow(nd[no][...,0],vmin=vmin,vmax=vmax,cmap=plt.cm.seismic,aspect="auto")
plt.show()