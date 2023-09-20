import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
true = np.load('../marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../marmousi_model/linear_vp.npy')[:,expand:-expand]
inverted = np.load('./results/paravpF03E49.npy')[npml:-npml, npml+expand:-npml-expand]

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

