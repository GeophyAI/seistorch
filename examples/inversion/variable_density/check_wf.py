import numpy as np
import matplotlib.pyplot as plt
import glob

bwidth = 50
ffiles = sorted(glob.glob('wf_pml/wf_foward*.npy'))
bfiles = sorted(glob.glob('wf_pml/wf_backward*.npy'))

ftime = 1500
btime = 2000-ftime-3

fsnapshot = np.load(ffiles[ftime])[0][bwidth:-bwidth,bwidth:-bwidth]
bsnapshot = np.load(bfiles[btime])[0][bwidth:-bwidth,bwidth:-bwidth]

fig, axes= plt.subplots(1,3,figsize=(10,3))
vmin, vmax = np.percentile(fsnapshot, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
axes[0].imshow(fsnapshot, **kwargs)
axes[0].set_title('Forward')
axes[1].imshow(bsnapshot, **kwargs)
axes[1].set_title('Backward')
axes[2].imshow(fsnapshot-bsnapshot, **kwargs)
axes[2].set_title('Error')
plt.show()