import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
freq = 2
epoch = 49
true = np.load('../marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../marmousi_model/linear_vp.npy')[:,expand:-expand]
invt_cg = np.load(f'./results/towed_cg/paravpF{freq:02d}E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
invt_sd = np.load(f'./results/towed_sd/paravpF{freq:02d}E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
invt_adam = np.load(f'./results/towed_adam/paravpF{freq:02d}E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(3,2,figsize=(12,8))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0,0].imshow(true,**kwargs)
axes[0,0].set_title("True")
axes[0,1].imshow(init,**kwargs)
axes[0,1].set_title("Initial")
axes[1,0].imshow(invt_cg,**kwargs)
axes[1,0].set_title("CG")
axes[1,1].imshow(invt_sd,**kwargs)
axes[1,1].set_title("SD")
axes[2,0].imshow(invt_adam,**kwargs)
axes[2,0].set_title("Adam")

trace = 450
axes[2,1].plot(true[:,trace],label="True")
axes[2,1].plot(init[:,trace],label="Initial")
axes[2,1].plot(invt_cg[:,trace],label="CG")
axes[2,1].plot(invt_sd[:,trace],label="SD")
axes[2,1].plot(invt_adam[:,trace],label="Adam")

axes[2,1].legend()
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)