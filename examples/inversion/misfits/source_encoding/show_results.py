import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
loss = "l2_1shot"
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
inverted = np.load(f'./{loss}/paravpF00E00.npy')[npml:-npml, npml+expand:-npml-expand]
grad = np.load(f'./{loss}/gradvpF00E00.npy')[npml:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(3,1,figsize=(8,10))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True")
axes[1].imshow(init,**kwargs)
axes[1].set_title("Initial")
axes[2].imshow(inverted,**kwargs)
axes[2].set_title("Inverted")
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)

fig, ax=plt.subplots(1,1,figsize=(5,4))
vmin,vmax=grad.min(),grad.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
ax.imshow(grad,**kwargs)
ax.set_title("Gradient")
plt.tight_layout()
plt.savefig(f"Gradient_{loss}.png",dpi=300)
