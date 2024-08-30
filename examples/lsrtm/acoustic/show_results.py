import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import laplace

os.makedirs('figures', exist_ok=True)
rootpath = r'./results_traditional_cg_born'

F = 0
E = 30
dh = 20
bwidth = 50
expand = 50

# ref = torch.load(f'{rootpath}/model_F{F:02d}E{E:02d}.pt', map_location='cpu')['m']
# grad = torch.load(f'{rootpath}/grad_m_F{F:02d}E{E:02d}.pt', map_location='cpu')

ref = torch.load(f'{rootpath}/model_{E}.pt', map_location='cpu')['m']
grad = torch.load(f'{rootpath}/grad_m_{E}.pt', map_location='cpu')

# ref = laplace(ref, mode='reflect')

ref = ref[bwidth:-bwidth, bwidth+expand:-expand-bwidth]
grad = grad[bwidth:-bwidth, bwidth+expand:-expand-bwidth]
extent = [0, ref.shape[1]*dh, ref.shape[0]*dh, 0]
fig, ax = plt.subplots(1,1,figsize=(5,3))
vmin, vmax=np.percentile(ref, [0, 100])
ax.imshow(ref, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
ax.set_title('Inverted')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Depth (m)')
plt.tight_layout()
fig.savefig('figures/inverted.png', dpi=300, bbox_inches='tight')
plt.show()

true = np.load('./velocity/true_vp.npy')[:,expand:-expand]
init = np.load('./velocity/smooth_vp.npy')[:,expand:-expand]
delta_m = 2*(true-init)/init

trace_no = 200
fig, ax = plt.subplots(1,1,figsize=(5,3))
zz = np.arange(ref.shape[0])*dh
plt.plot(zz, ref[:,trace_no], 'r', label='Inverted')
plt.plot(zz, delta_m[:,trace_no], 'b', label='True')
plt.xlabel('Depth (m)')
plt.legend()
plt.tight_layout()
fig.savefig('figures/trace.png', dpi=300, bbox_inches='tight')
plt.show()

# fig, ax = plt.subplots(1,1,figsize=(5,3))
# vmin, vmax=np.percentile(grad, [2, 98])
# # add colorbar
# cbar = plt.colorbar(ax.imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto'))
# plt.tight_layout()
# fig.savefig(f'figures/grad.png', dpi=300, bbox_inches='tight')
# plt.show()




