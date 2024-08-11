import numpy as np
import matplotlib.pyplot as plt
import torch, os, glob

bwidth = 50
dh=10 # grid spacing
folders = glob.glob('results/w1d_*')
grads = []
for folder in folders:
    grad = torch.load(f'{folder}/grad_vp_0.pt').cpu().numpy()
    grad = grad[bwidth:-bwidth, bwidth:-bwidth]
    grads.append(grad)
fig, axes = plt.subplots(2, (len(grads)//2), figsize=(10, 7))
extent = [0, grad.shape[1] * dh, grad.shape[0] * dh, 0]
for i, (ax, grad) in enumerate(zip(axes.ravel(), grads)):
    vmin, vmax = np.percentile(grad, [1, 99])
    kwargs = dict(vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
    ax.imshow(grad, **kwargs)
    ax.set_title(f'{folders[i]}')
    ax.axis('off')
    circle = plt.Circle((grad.shape[1]//2 * dh, grad.shape[0]//2 * dh), 600, fill=False, color='r')
    ax.add_artist(circle)

plt.tight_layout()
plt.savefig(f'figures/Gradient_diff_norm.png', dpi=300, bbox_inches='tight')