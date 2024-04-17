import numpy as np
import matplotlib.pyplot as plt
import torch
# Load the model
root = r'/home/shaowinw/seistorch/examples/inversion/misfits/travel_time_misfit/results/traveltime'

epoch = 7
grad = torch.load(f'{root}/grad_vp_{epoch}.pt').cpu().detach().numpy()

grad = grad[50:-50, 50:-50]

fig, ax=plt.subplots(figsize=(8, 4))
vmin, vmax=np.percentile(grad, [3, 97])
plt.imshow(grad, 
           cmap='seismic', 
           aspect='auto',
           vmin=vmin, 
           vmax=vmax)
plt.colorbar()
plt.show()