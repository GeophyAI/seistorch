import numpy as np
import matplotlib.pyplot as plt
import torch

rootpath = r'/home/shaowinw/seistorch/examples/newversion/marmousi/results/habc_inv_habc'
btype = 'habc'
epoch = 199
expand = 50
pml = 50
vel = torch.load(f'{rootpath}/model_{epoch}.pt', map_location='cpu')['vp']
grad = torch.load(f'{rootpath}/grad_vp_{epoch}.pt', map_location='cpu')

vel = vel[expand:-expand, expand+pml:-expand-pml]
grad = grad[expand:-expand, expand+pml:-expand-pml]
print(vel.shape, grad.shape)
fig, ax = plt.subplots(1,1,figsize=(5,3))
ax.imshow(vel, vmin=1500, vmax=5500, cmap='seismic', aspect='auto')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1,figsize=(5,3))
vmin, vmax=np.percentile(grad, [2, 98])
ax.imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.tight_layout()
plt.show()



