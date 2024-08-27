import os
import torch
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

dh = 10
bwidth = 50
vti = torch.load(f'results/vti_cg_bornobs/model_29.pt', map_location='cpu')['m'][bwidth:-bwidth, bwidth:-bwidth]
tti = torch.load(f'results/tti_cg_bornobs/model_29.pt', map_location='cpu')['m'][bwidth:-bwidth, bwidth:-bwidth]
true = np.load('models/true_m.npy')
extent = [0, vti.shape[1]*dh, vti.shape[0]*dh, 0]

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
vmin, vmax=np.percentile(true, [1, 99])
for ax, m, title in zip(axes, [true, vti, tti], ['True', 'VTI', 'TTI', ]):
    ax.imshow(m, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
    ax.set_title(title)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
plt.tight_layout()
plt.savefig('figures/compare_vti_tti.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
trace_no = 200
zz = np.arange(vti.shape[0])*dh
plt.plot(zz, vti[:,trace_no], 'r', label='VTI')
plt.plot(zz, tti[:,trace_no], 'b', label='TTI')
plt.plot(zz, true[:,trace_no], 'g', label='True')
plt.xlabel('Depth (m)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/compare_vti_tti_trace.png', dpi=100, bbox_inches='tight')
plt.show()

