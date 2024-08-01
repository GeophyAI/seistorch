import numpy as np
import matplotlib.pyplot as plt
import torch
npml = 50
expand = 50
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
inverted_l2 = torch.load('./results/towed_l2/model_99.pt')['vp'].cpu().numpy()[npml:-npml, npml:-npml]
inverted_cs = torch.load('./results/towed_cs/model_99.pt')['vp'].cpu().numpy()[npml:-npml, npml:-npml]


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



fig, ax=plt.subplots(1,1,figsize=(5,3))
trace_no = 250
z = np.arange(true.shape[0])*20
ax.plot(z, inverted_l2[:,trace_no], label="L2 Inverted")
ax.plot(z, inverted_cs[:,trace_no], label="CS Inverted")
ax.plot(z, true[:,trace_no], label="True")
ax.plot(z, init[:,trace_no], label="Initial")
ax.legend()
ax.set_title("Trace")
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Velocity (m/s)")
plt.tight_layout()

plt.savefig("Trace.png",dpi=300, bbox_inches='tight')
