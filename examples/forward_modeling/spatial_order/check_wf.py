import numpy as np
import matplotlib.pyplot as plt

time = 1000
pmln = 50
wfo2 = np.load(f'wf_pml_o2/wf_foward{time:04d}.npy')[0][pmln:-pmln, pmln:-pmln]
wfo4 = np.load(f'wf_pml_o4/wf_foward{time:04d}.npy')[0][pmln:-pmln, pmln:-pmln]
wfo6 = np.load(f'wf_pml_o6/wf_foward{time:04d}.npy')[0][pmln:-pmln, pmln:-pmln]

vmin, vmax=np.percentile(wfo2, [1, 99])
fig, axes= plt.subplots(1, 3, figsize=(9, 3))
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[0].imshow(wfo2, **kwargs)
axes[0].set_title('Spatial order = 2')
axes[1].imshow(wfo4, **kwargs)
axes[1].set_title('Spatial order = 4')
axes[2].imshow(wfo6, **kwargs)
axes[2].set_title('Spatial order = 6')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('Spatial_order.png')
plt.show()

