import torch
import numpy as np
import matplotlib.pyplot as plt

epoch = 199
npml = 50
expand = 50

model = np.load(f'./results/towed_jax/inverted{epoch:03d}.npy')[0]
grad = np.load(f'./results/towed_jax/gradient{epoch:03d}.npy')[0]

vp = model[npml:-npml, npml+expand:-npml-expand]
grad = grad[npml:-npml, npml+expand:-npml-expand]

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

