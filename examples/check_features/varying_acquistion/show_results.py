import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
inverted = np.load('./results/towed/paravpF00E00.npy')[npml:-npml, npml+expand:-npml-expand]
grad = np.load('./results/towed/gradvpF00E00.npy')[npml:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(4,1,figsize=(8,12))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True")
axes[1].imshow(init,**kwargs)
axes[1].set_title("Initial")
axes[2].imshow(inverted,**kwargs)
axes[2].set_title("Inverted")
axes[3].plot(inverted[:,300], label="Inverted")
axes[3].plot(true[:,300], label="True")
axes[3].plot(init[:,300], label="Initial")
axes[3].legend()
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)

fig, axes=plt.subplots(1,1,figsize=(5,4))
vmin,vmax=np.percentile(grad,[2,98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes.imshow(grad,**kwargs)
axes.set_title("Gradient")
plt.tight_layout()
plt.savefig("Gradient.png",dpi=300)

