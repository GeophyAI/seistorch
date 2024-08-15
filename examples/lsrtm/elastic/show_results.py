import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import laplace
os.makedirs('./figures', exist_ok=True)
rootpath = r'./results'

F = 0
E = 49
bwidth = 50
expand = 50

# m_vp = torch.load(f'{rootpath}/model_F{F:02d}E{E:02d}.pt', map_location='cpu')['m']
# grad = torch.load(f'{rootpath}/grad_m_F{F:02d}E{E:02d}.pt', map_location='cpu')

true_mvp = np.load(f'./velocity/true_mvp.npy')[:,expand:-expand]
true_mvs = np.load(f'./velocity/true_mvs.npy')[:,expand:-expand]

m = torch.load(f'{rootpath}/model_{E}.pt', map_location='cpu')
m_vp = m['rvp']
m_vs = m['rvs']
# m_vp = laplace(m_vp, mode='reflect')
dh = 20
m_vp = m_vp[bwidth:-bwidth, bwidth+expand:-expand-bwidth]
m_vs = m_vs[bwidth:-bwidth, bwidth+expand:-expand-bwidth]

fig, axes = plt.subplots(1,2,figsize=(8,3))
vmin, vmax=np.percentile(true_mvp, [5, 95])
extent = [0, m_vp.shape[1]*dh, m_vp.shape[0]*dh, 0]
axes[0].imshow(m_vp, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
axes[0].set_title(f'Inverted vp')
vmin, vmax = np.percentile(true_mvs, [5, 95])
axes[1].imshow(m_vs, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
axes[1].set_title(f'Inverted vs')
for ax in axes.ravel():
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
plt.tight_layout()
fig.savefig(f'figures/inverted.png', dpi=300, bbox_inches='tight')
plt.show()


trace_no = 200
fig,axes = plt.subplots(1,2,figsize=(8,3))
zz = np.arange(m_vp.shape[0])*dh
axes[0].plot(zz, m_vp[:,trace_no], 'r', label=f'Inverted rvp')
axes[0].plot(zz, true_mvp[:,trace_no], 'b', label=f'True rvp')
axes[1].plot(zz, m_vs[:,trace_no], 'r', label=f'Inverted rvs')
axes[1].plot(zz, true_mvs[:,trace_no], 'b', label=f'True rvs')
for ax in axes.ravel():
    ax.set_xlabel('Depth (m)')
    ax.legend()
    ax.set_title(f'Profile at {trace_no*dh}m')
plt.tight_layout()
fig.savefig(f'figures/trace.png', dpi=300, bbox_inches='tight')
plt.show()


