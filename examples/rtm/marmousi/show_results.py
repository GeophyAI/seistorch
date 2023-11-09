import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

npml = 50
expand = 50
grad_true = np.load("./results/rtm_true/gradvpF00E00.npy")[npml:-npml, npml+expand:-npml-expand]
grad_init = np.load("./results/rtm_true/gradvpF00E00.npy")[npml:-npml, npml+expand:-npml-expand]
true = np.load("../models/marmousi_model/smooth_true_vp_for_rtm.npy")[:,expand:-expand]
init = np.load("../../../models/marmousi_model/linear_vp.npy")[:,expand:-expand]

seabed = np.load("../models/marmousi_model/seabed.npy")

# RTM image = grad(vp) * vp^3

rtm_true = grad_true * seabed[:,expand:-expand]*true**3
rtm_init = grad_init * seabed[:,expand:-expand]*init**3

laplace(rtm_true, rtm_true)
laplace(rtm_init, rtm_init)

# laplace(grad_true, grad_true)
# laplace(grad_init, grad_init)

# Show velocity models
fig, axes=plt.subplots(1,2,figsize=(8,3))
vmin, vmax=np.percentile(grad_true, [5, 95])
kwargs={"cmap":"gray","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(grad_true*seabed[:,expand:-expand],**kwargs)
axes[0].set_title("Gradient with True model")
axes[1].imshow(grad_init*seabed[:,expand:-expand],**kwargs)
axes[1].set_title("Gradient Initial model")
plt.tight_layout()
plt.show()
fig.savefig("Gradients.png",dpi=300)
# Show gradients
fig, axes=plt.subplots(1,2,figsize=(8,3))
kwargs={"cmap":"seismic","aspect":"auto","vmin":1500,"vmax":5500}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True model")
axes[1].imshow(init,**kwargs)
axes[1].set_title("Initial model")
plt.tight_layout()
plt.show()
fig.savefig("Velocity.png",dpi=300)

# Show reverse time migration images
fig, axes=plt.subplots(1,2,figsize=(8,3))
vmin, vmax=np.percentile(rtm_true, [2, 98])
kwargs={"cmap":"gray","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(rtm_true,**kwargs)
axes[0].set_title("Reverse Time Migration with True model")
axes[1].imshow(rtm_init,**kwargs)
axes[1].set_title("Reverse Time Migration with Initial model")
plt.tight_layout()
plt.show()
fig.savefig("RTM.png",dpi=300)


