import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
inverted_l2 = np.load('./results/towed_l2/paravpF03E24.npy')[npml:-npml, npml:-npml]
inverted_cs = np.load('./results/towed_cs/paravpF03E24.npy')[npml:-npml, npml:-npml]


import os
savepath = r"inverted"
os.makedirs(savepath, exist_ok=True)
np.save(os.path.join(savepath, "inverted_l2.npy"), inverted_l2)
np.save(os.path.join(savepath, "inverted_cs.npy"), inverted_cs)

inverted_l2 = inverted_l2[:,expand:-expand]
inverted_cs = inverted_cs[:,expand:-expand]

fig, axes=plt.subplots(2,2,figsize=(8,6))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0,0].imshow(true,**kwargs)
axes[0,0].set_title("True")
axes[0,1].imshow(init,**kwargs)
axes[0,1].set_title("Initial")
axes[1,0].imshow(inverted_l2,**kwargs)
axes[1,0].set_title("L2 Inverted")
axes[1,1].imshow(inverted_cs,**kwargs)
axes[1,1].set_title("CS Inverted")
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)



fig, ax=plt.subplots(1,1,figsize=(5,4))

ax.plot(inverted_l2[:,300], label="L2 Inverted")
ax.plot(inverted_cs[:,300], label="CS Inverted")
ax.plot(true[:,300], label="True")
ax.plot(init[:,300], label="Initial")
ax.legend()
ax.set_title("Trace")
plt.tight_layout()
plt.savefig("Trace.png",dpi=300)
