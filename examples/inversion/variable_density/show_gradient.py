import numpy as np
import matplotlib.pyplot as plt
import torch

bwidth = 50

model = 'vp'

AD=torch.load(f'AD/grad_{model}_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
BS=torch.load(f'BS/grad_{model}_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]

fig, axes= plt.subplots(1,2,figsize=(10,5))
vmin, vmax = np.percentile(AD, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
axes[0].imshow(AD, **kwargs)
axes[0].set_title('AD')
axes[1].imshow(BS, **kwargs)
axes[1].set_title('BS')
plt.show()

