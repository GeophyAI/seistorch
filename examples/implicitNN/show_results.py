import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
true = np.load('../marmousi_model_half/true_vp.npy')[:,expand:-expand]
init = np.load('../marmousi_model_half/linear_vp.npy')[:,expand:-expand]
# inverted = np.load('./results_explicit/paravpF00E99.npy')[npml:-npml, npml+expand:-npml-expand]
inverted = np.load('./results_implicit_l2/paravpF00E499.npy')[npml:-npml, npml+expand:-npml-expand]

loss = np.load("./results_implicit_l2/loss.npy")
plt.plot(loss.flatten())
plt.legend()
plt.show()

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

trace = 100
fig, ax=plt.subplots(1,1,figsize=(6,4))
ax.plot(true[:,trace],label="True")
ax.plot(init[:,trace],label="Initial")
ax.plot(inverted[:,trace],label="Inverted")
ax.legend()
fig.savefig("Inverted_profile.png",dpi=300)
plt.show()

np.save("ipinverted.npy", np.pad(inverted, ((0, 0), (npml, npml)), 'edge'))