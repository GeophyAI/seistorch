import numpy as np
import matplotlib.pyplot as plt
import torch, os

dh = 10
bwidth = 50
envelope = torch.load('results/envelope/grad_vp_0.pt').cpu().numpy()
l2 = torch.load('results/l2/grad_vp_0.pt').cpu().numpy()
w1d = torch.load('results/w1d_envelope/grad_vp_0.pt').cpu().numpy()
savepath = 'figures'
os.makedirs(savepath, exist_ok=True)    
envelope = envelope[bwidth:-bwidth, bwidth:-bwidth]
l2 = l2[bwidth:-bwidth, bwidth:-bwidth]
w1d = w1d[bwidth:-bwidth, bwidth:-bwidth]
extent = [0, l2.shape[1] * dh, l2.shape[0] * dh, 0]
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
vmin, vmax = np.percentile(l2, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
axes[0].set_title('L2 Gradient')
plt.colorbar(axes[0].imshow(l2, **kwargs), ax=axes[0], orientation='vertical')

vmin, vmax = np.percentile(envelope, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
axes[1].imshow(envelope, **kwargs)
axes[1].set_title('Envelope Gradient')
plt.colorbar(axes[1].imshow(envelope, **kwargs), ax=axes[1], orientation='vertical')

vmin, vmax = np.percentile(w1d, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
axes[2].imshow(w1d, **kwargs)
axes[2].set_title('W1d Gradient')
plt.colorbar(axes[2].imshow(w1d, **kwargs), ax=axes[2], orientation='vertical')

for ax in axes.ravel():
    # Draw a circle centered at the middle of the model with radius 600m
    circle = plt.Circle((l2.shape[1]//2 * dh, l2.shape[0]//2 * dh), 600, fill=False, color='r')
    ax.add_artist(circle)

plt.tight_layout()
plt.savefig(f'{savepath}/Gradient_vs_l2.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax=plt.subplots(1, 1, figsize=(5, 3))
x = np.arange(0, l2.shape[1], 1) * dh
def norm(d):
    return d/np.abs(d).max()
ax.plot(x, norm(envelope[envelope.shape[0]//2, :]), label='Envelope')
ax.plot(x, norm(l2[l2.shape[0]//2, :]), label='L2')
ax.plot(x, norm(w1d[w1d.shape[0]//2, :]), label='W1d')
# Draw the boundary of the circle
ax.axvline(x=l2.shape[1]//2*dh - 600, color='black', label='Boundary L')
ax.axvline(x=l2.shape[1]//2*dh + 600, color='black', label='Boundary R')
ax.legend()
ax.set_xlabel('x (m)')
ax.set_title('Gradient at depth 500m')
plt.savefig(f'{savepath}/Gradient_line.png', dpi=300, bbox_inches='tight')
plt.show()