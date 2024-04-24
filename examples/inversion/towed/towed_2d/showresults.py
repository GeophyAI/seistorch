import torch
import numpy as np
import matplotlib.pyplot as plt

epoch = 80
npml = 50
expand = 50

model = torch.load(f'./results/towed/model_{epoch}.pt', map_location='cpu')
grad = torch.load(f'./results/towed/grad_vp_{epoch}.pt', map_location='cpu')

vp = model['vp'].numpy()[npml:-npml, npml+expand:-npml-expand]
grad = grad.numpy()[npml:-npml, npml+expand:-npml-expand]

kwargs = {'vmin':1500, 'vmax':5500, 'cmap':'seismic', 'aspect': 'auto'}

plt.figure(figsize=(8, 3))
plt.imshow(vp, **kwargs)
plt.colorbar()
plt.show()

vmin, vmax = np.percentile(grad, [2, 98])
plt.figure(figsize=(8, 3))
plt.imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.show()

