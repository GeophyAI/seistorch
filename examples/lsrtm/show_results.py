import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import laplace

rootpath = r'./results_traditional_adam'

F = 0
E = 19
bwidth = 50
expand = 50

# vel = torch.load(f'{rootpath}/model_F{F:02d}E{E:02d}.pt', map_location='cpu')['m']
# grad = torch.load(f'{rootpath}/grad_m_F{F:02d}E{E:02d}.pt', map_location='cpu')

vel = torch.load(f'{rootpath}/model_{E}.pt', map_location='cpu')['m']
grad = torch.load(f'{rootpath}/grad_m_{E}.pt', map_location='cpu')

# vel = laplace(vel, mode='reflect')

vel = vel[bwidth:-bwidth, bwidth+expand:-expand-bwidth]
grad = grad[bwidth:-bwidth, bwidth+expand:-expand-bwidth]
print(vel.max(), vel.min())
print(grad.max(), grad.min())
fig, ax = plt.subplots(1,1,figsize=(5,3))
vmin, vmax=np.percentile(vel, [0, 100])
ax.imshow(vel, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
plt.tight_layout()
plt.show()

true = np.load('./velocity/true_vp.npy')[:,expand:-expand]
init = np.load('./velocity/smooth_vp.npy')[:,expand:-expand]
delta_m = 2*(true-init)/init

trace_no = 120
vel[0:24] = 0.
plt.plot(vel[:,trace_no], label='Inverted')
plt.plot(delta_m[:,trace_no], label='True')
plt.legend()
plt.show()
fig.savefig(f'vel.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(5,3))
vmin, vmax=np.percentile(grad, [2, 98])
# add colorbar
cbar = plt.colorbar(ax.imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto'))
plt.tight_layout()

fig.savefig(f'grad.png', dpi=300, bbox_inches='tight')





