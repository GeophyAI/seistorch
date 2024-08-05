import numpy as np
import matplotlib.pyplot as plt
import torch

bwidth = 50

models = ['vp', 'rx', 'rz']
tasks = ['acoustic_habc_sm', 
         'acoustic_habc_bg', 
         'acoustic_fwim_habc_sm', 
         'acoustic_fwim_habc_zero']

# Acoustic case
sm = torch.load(f'kernels/acoustic_habc_sm/grad_vp_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
bg = torch.load(f'kernels/acoustic_habc_bg/grad_vp_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]

fig, axes=plt.subplots(1,2,figsize=(6,2))
vmin, vmax = np.percentile(sm, [2, 98])
axes[0].imshow(sm, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[0].set_title('vp kernel with smooth model')
vmin, vmax = np.percentile(bg, [2, 98])
axes[1].imshow(bg, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[1].set_title('vp kernel with background model')
axes[0].axis('off')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# FWIM case
# vp, rx, rz all smoothed
vp = torch.load(f'kernels/acoustic_fwim_habc_sm/grad_vp_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
rx = torch.load(f'kernels/acoustic_fwim_habc_sm/grad_rx_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
rz = torch.load(f'kernels/acoustic_fwim_habc_sm/grad_rz_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
fig, axes = plt.subplots(1,3,figsize=(9,2))
vmin, vmax = np.percentile(vp, [2, 98])
axes[0].imshow(vp, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[0].set_title('vp kernel with smoothed model')
vmin, vmax = np.percentile(rx, [2, 98])
axes[1].imshow(rx, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[1].set_title('rx kernel with smoothed model')
vmin, vmax = np.percentile(rz, [2, 98])
axes[2].imshow(rz, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[2].set_title('rz kernel with smoothed model')
axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')
plt.tight_layout()
plt.show()


# FWIM case
# vp smoothed, rx rz are zero
vp = torch.load(f'kernels/acoustic_fwim_habc_zero/grad_vp_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
rx = torch.load(f'kernels/acoustic_fwim_habc_zero/grad_rx_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
rz = torch.load(f'kernels/acoustic_fwim_habc_zero/grad_rz_0.pt').cpu().detach().numpy()[bwidth:-bwidth,bwidth:-bwidth]
fig, axes = plt.subplots(1,3,figsize=(9,2))
vmin, vmax = np.percentile(vp, [2, 98])
axes[0].imshow(vp, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[0].set_title('vp kernel with zero ref')
vmin, vmax = np.percentile(rx, [2, 98])
axes[1].imshow(rx, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[1].set_title('rx kernel with zero ref')
vmin, vmax = np.percentile(rz, [2, 98])
axes[2].imshow(rz, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
axes[2].set_title('rz kernel with zero ref')
axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')
plt.tight_layout()
plt.show()

