import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import laplace

os.makedirs('figures', exist_ok=True)
rootpath = r'./results/vti_cg_bornobs'
basename = rootpath.split('/')[-1]
F = 0
E = 29
dh = 10
bwidth = 50

ref = torch.load(f'{rootpath}/model_{E}.pt', map_location='cpu')['m']
grad = torch.load(f'{rootpath}/grad_m_{E}.pt', map_location='cpu')

# ref = laplace(ref, mode='reflect')

ref = ref[bwidth:-bwidth, bwidth:-bwidth]
grad = grad[bwidth:-bwidth, bwidth:-bwidth]
extent = [0, ref.shape[1]*dh, ref.shape[0]*dh, 0]
fig, ax = plt.subplots(1,1,figsize=(5,3))
vmin, vmax=np.percentile(ref, [0, 100])
ax.imshow(ref, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
ax.set_title('Inverted')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Depth (m)')
ax.set_title('Inverted')
plt.tight_layout()
fig.savefig(f'figures/inverted_{basename}.png', dpi=300, bbox_inches='tight')
plt.show()

true = np.load('./models/vp.npy')
init = np.load('./models/vp_smooth.npy')
delta_m = 2*(true-init)/init

trace_no = 200
fig, ax = plt.subplots(1,1,figsize=(5,3))
zz = np.arange(ref.shape[0])*dh
plt.plot(zz, ref[:,trace_no], 'r', label='Inverted')
plt.plot(zz, delta_m[:,trace_no], 'b', label='True')
plt.xlabel('Depth (m)')
plt.legend()
plt.tight_layout()
fig.savefig(f'figures/trace_{basename}.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1,1,figsize=(5,3))
vmin, vmax=np.percentile(grad, [2, 98])
# add colorbar
cbar = plt.colorbar(ax.imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto'))
plt.tight_layout()
fig.savefig(f'figures/grad.png', dpi=300, bbox_inches='tight')
ax.set_title('Gradient')
plt.show()




