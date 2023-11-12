import numpy as np
import matplotlib.pyplot as plt

npml = 50
epoch = 49
true = np.load('./velocity_model/true.npy')
init = np.load('./velocity_model/init.npy')
inverted = np.load(f'./results_w1d/paravpF00E{epoch:02d}.npy')[npml:-npml, npml:-npml]
grad = np.load(f'./results_w1d/gradvpF00E{epoch:02d}.npy')[npml:-npml, npml:-npml]

fig, axes=plt.subplots(1,4,figsize=(12,3))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True")
axes[1].imshow(init,**kwargs)
axes[1].set_title("Initial")
axes[2].imshow(inverted,**kwargs)
axes[2].set_title("Inverted")

trace = 100
axes[3].plot(true[:,trace],label="True")
axes[3].plot(init[:,trace],label="Initial")
axes[3].plot(inverted[:,trace],label="Inverted Vertical Profile")
axes[3].plot(inverted[trace,:],label="Inverted Horizontal Profile")
axes[3].legend()

plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)
plt.show()



vmin, vmax = np.percentile(grad, [2, 98])
plt.imshow(grad, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title("Gradient")
plt.tight_layout()
plt.savefig("Gradient.png",dpi=300)


