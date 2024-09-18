import numpy as np
import matplotlib.pyplot as plt

bwidth = 50
invt = np.load('results/vti_bornobs_jax/model_F00E29.npy')[-1][bwidth:-bwidth, bwidth:-bwidth]
print(invt.max(), invt.min())
true = np.load('./models/vp.npy')
init = np.load('./models/vp_smooth.npy')
true_m = 2*(true-init)/init

vmin, vmax=np.percentile(true_m, [2, 98])

fig, axes= plt.subplots(1,2,figsize=(10,3))
axes[0].imshow(invt, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[0].set_title('Inverted')
axes[1].imshow(true_m, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[1].set_title('True')
plt.tight_layout()
plt.show()

fig,ax=plt.subplots(1,1,figsize=(5,3))
plt.plot(invt[:, 100], label='Inverted')
plt.plot(true_m[:, 100], label='True')
plt.legend()
plt.tight_layout()
plt.show()

# grad = np.load('results/vti_bornobs_jax/gradient_F00E01.npy')[3][bwidth:-bwidth, bwidth:-bwidth]
# vmin, vmax=np.percentile(grad, [2, 98])
# print(grad.max(), grad.min())
# fig, ax = plt.subplots(1,1,figsize=(5,3))
# ax.imshow(grad, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
# ax.set_title('Gradient')
# plt.tight_layout()
# plt.show()
