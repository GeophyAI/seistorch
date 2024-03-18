import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
inv_without_mul = np.load('./without_multiples/results/paravpF03E49.npy')[npml:-npml, npml+expand:-npml-expand]
inv_with_mul = np.load('./with_multiples/results/paravpF03E49.npy')[:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(2,2,figsize=(8,6))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0,0].imshow(true,**kwargs)
axes[0,0].set_title("True")
axes[0,1].imshow(init,**kwargs)
axes[0,1].set_title("Initial")
axes[1,0].imshow(inv_without_mul,**kwargs)
axes[1,0].set_title("Inverted without multiples")
axes[1,1].imshow(inv_with_mul,**kwargs)
axes[1,1].set_title("Inverted with multiples")
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)