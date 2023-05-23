import matplotlib.pyplot as plt
import numpy as np

PMLN = 50
grad = np.load("/mnt/data/wangsw/inversion/marmousi_10m/acoustic1st/wd/grad_params_vp.npy")[PMLN:-PMLN, PMLN:-PMLN]
grad2 = np.load("/mnt/data/wangsw/inversion/marmousi_10m/acoustic1st/wd/gradvpF00E00.npy")[PMLN:-PMLN, PMLN:-PMLN]
print(grad.shape)

fig, axes = plt.subplots(1,2, figsize=(10,5))
vmin, vmax=np.percentile(grad, [2, 98])
ax0=axes[0].imshow(grad, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
ax1=axes[1].imshow(grad2, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
plt.colorbar(ax0)
plt.colorbar(ax1)
plt.show()