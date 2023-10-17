import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
pmln = 50
expand = 50
true = np.load("../marmousi_model/true_vp.npy")[:,expand:-expand]
init = np.load("../marmousi_model/linear_vp.npy")[:,expand:-expand]

grad3d_true = np.load('results/rtm_true/grad3d.npy')
grad3d_true = grad3d_true[:,:, 50:-50, 100:-100].squeeze()

grad3d_init = np.load('results/rtm_init/grad3d.npy')
grad3d_init= grad3d_init[:,:, 50:-50, 100:-100].squeeze()

for i in range(grad3d_true.shape[0]):
    laplace(grad3d_true[i], grad3d_true[i])
    laplace(grad3d_init[i], grad3d_init[i])

grad3d_init *= init[np.newaxis, :, :]**3
grad3d_true *= true[np.newaxis, :, :]**3

grad3d_init[:,0:24] = 0
grad3d_true[:,0:24] = 0

fig, axes=plt.subplots(2,1,figsize=(6,4))
vmin, vmax=np.percentile(np.sum(grad3d_true, 0), [2, 98])
kwargs={"cmap":"gray","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(np.sum(grad3d_true, 0),**kwargs)
axes[0].set_title("RTM with True model")
axes[1].imshow(np.sum(grad3d_init, 0),**kwargs)
axes[1].set_title("RTM with Initial model")
plt.tight_layout()
plt.show()

trace = 200
ig_true = grad3d_true[:,:,trace]
ig_init = grad3d_init[:,:,trace]

plt.imshow(true, cmap='seismic', vmin=1500, vmax=5500)
plt.vlines(trace, 0, true.shape[0], color='r')
plt.show()
# laplace(ig, ig)

fig, axes=plt.subplots(1,2,figsize=(6,4))
vmin, vmax=np.percentile(ig_true, [2, 98])
kwargs={"cmap":"gray","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(ig_true.T,**kwargs)
axes[0].set_title("CIG with True model")
axes[1].imshow(ig_init.T,**kwargs)
axes[1].set_title("CIG with Initial model")
plt.tight_layout()
fig.savefig("CIG.png",dpi=300)
plt.show()