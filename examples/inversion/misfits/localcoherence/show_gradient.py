import numpy as np
import matplotlib.pyplot as plt
import torch

dh = 10
bwidth = 50
envelope = torch.load('results/lc/grad_vp_0.pt').cpu().numpy()
l2 = torch.load('results/l2/grad_vp_0.pt').cpu().numpy()

envelope = envelope[bwidth:-bwidth, bwidth:-bwidth]
l2 = l2[bwidth:-bwidth, bwidth:-bwidth]
extent = [0, l2.shape[1] * dh, l2.shape[0] * dh, 0]
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
vmin, vmax = np.percentile(l2, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
axes[0].set_title('L2 Gradient')
plt.colorbar(axes[0].imshow(l2, **kwargs), ax=axes[0], orientation='vertical')

vmin, vmax = np.percentile(envelope, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
axes[1].imshow(envelope, **kwargs)
axes[1].set_title('LC Gradient')
plt.colorbar(axes[1].imshow(envelope, **kwargs), ax=axes[1], orientation='vertical')

for ax in axes.ravel():
    # Draw a circle centered at the middle of the model with radius 150m
    circle = plt.Circle((l2.shape[1]//2 * dh, l2.shape[0]//2 * dh), 150, fill=False, color='r')
    ax.add_artist(circle)

plt.tight_layout()
plt.savefig('figures/Gradient.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax=plt.subplots(1, 1, figsize=(5, 3))
x = np.arange(0, l2.shape[1], 1) * dh
ax.plot(x, envelope[envelope.shape[0]//2, :], label='LC')
ax.plot(x, l2[l2.shape[0]//2, :], label='L2')
# Draw the boundary of the circle
ax.axvline(x=l2.shape[1]//2*dh - 150, color='black', label='Boundary L')
ax.axvline(x=l2.shape[1]//2*dh + 150, color='black', label='Boundary R')
ax.legend()
ax.set_xlabel('x (m)')
ax.set_title('Gradient at depth 500m')
plt.savefig('figures/Gradient_line.png', dpi=300, bbox_inches='tight')
plt.show()