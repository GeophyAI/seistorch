import numpy as np
import matplotlib.pyplot as plt
import torch

npml = 50
expand = 50
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
invt_torch = torch.load('./results/torch/model_F03E49.pt')['vp'].cpu().detach()[npml:-npml, npml+expand:-npml-expand]
invt_jax = np.load('./results/jax/model_F03E49.npy')[0, npml:-npml, npml+expand:-npml-expand]
grad = torch.load('./results/torch/grad_vp_F00E49.pt').cpu().detach()[npml:-npml, npml+expand:-npml-expand]
fig, axes=plt.subplots(4,1,figsize=(8,10))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True")
axes[1].imshow(init,**kwargs)
axes[1].set_title("Initial")
axes[2].imshow(invt_torch,**kwargs)
axes[2].set_title("Torch")
axes[3].imshow(invt_jax,**kwargs)
axes[3].set_title("Jax")
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)

fig, ax = plt.subplots(1,1,figsize=(5,4))
vmin,vmax=np.percentile(grad,[2, 98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
ax.imshow(grad,**kwargs)
ax.set_title("Gradient")
plt.tight_layout()
plt.savefig("Gradient.png",dpi=300)
plt.show()
