import torch
import numpy as np
import matplotlib.pyplot as plt

epoch = 500
npml = 50
expand = 50

model = torch.load(f'./results/towed/model_{epoch}.pt', map_location='cpu')
true = np.load('/home/shaowinw/Desktop/seistorch/examples/models/marmousi_model/true_vp.npy')


vp = model.detach().numpy()[npml:-npml, npml+expand:-npml-expand]
true = true[:,expand:-expand]
print(vp.max(), vp.min())
kwargs = {'vmin':1500, 'vmax':5500, 'cmap':'seismic', 'aspect': 'auto'}

plt.figure(figsize=(8, 3))
plt.imshow(vp, **kwargs)
plt.colorbar()
plt.show()

trace_no = 300
plt.plot(vp[:, trace_no], label='inverted')
plt.plot(true[:, trace_no], label='true')
plt.legend()
plt.show()


