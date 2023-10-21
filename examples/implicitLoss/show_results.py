import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
true = np.load('../marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../marmousi_model/linear_vp.npy')[:,expand:-expand]
inv_l2 = np.load('./results_l2/paravpF03E49.npy')[npml:-npml, npml+expand:-npml-expand]
inv_ip = np.load('./results_ip/paravpF00E20.npy')[npml:-npml, npml+expand:-npml-expand]

loss_l2 = np.load("./results_ip/loss.npy")
loss_ip = np.load("./results_l2/loss.npy")

for i in range(loss_l2.shape[0]):
    loss_l2[i] /= loss_l2[i][0]
    loss_ip[i] /= loss_ip[i][0]

fig, ax=plt.subplots(1,1,figsize=(8,5))
ax.plot(loss_l2.flatten(), label="L2")
ax.plot(loss_ip.flatten(), label="Implicit")
ax.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
plt.tight_layout()
plt.savefig("Loss.png",dpi=300)
plt.show()

fig, axes=plt.subplots(2,2,figsize=(8,6))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0,0].imshow(true,**kwargs)
axes[0,0].set_title("True")
axes[0,1].imshow(init,**kwargs)
axes[0,1].set_title("Initial")
axes[1,0].imshow(inv_l2,**kwargs)
axes[1,0].set_title("Inverted")
axes[1,1].imshow(inv_ip,**kwargs)
axes[1,0].set_title("Inverted")
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)

trace = 150
fig, ax=plt.subplots(1,1,figsize=(8,5))
ax.plot(true[:,trace],label="True")
ax.plot(init[:,trace],label="Initial")
ax.plot(inv_l2[:,trace],label="Inverted by L2")
ax.plot(inv_ip[:,trace],label="Inverted by Implicit")
ax.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
plt.tight_layout()
fig.savefig("Inverted.png",dpi=300)
plt.show()