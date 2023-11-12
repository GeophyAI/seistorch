import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
F = 0
E = 0
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
rand_inverted = np.load(f'./results_rand/paravpF{F:02d}E{E:02d}.npy')#[npml:-npml, npml+expand:-npml-expand]
pml_inverted = np.load(f'./results_pml/paravpF{F:02d}E{E:02d}.npy')[0:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(1,1,figsize=(6,3))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
plt.imshow(rand_inverted, **kwargs)
plt.title("With random boundary")
plt.show()

# Cut the boundary
rand_inverted = rand_inverted[0:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(4,1,figsize=(6,10))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True")
axes[1].imshow(init,**kwargs)
axes[1].set_title("Initial")
axes[2].imshow(rand_inverted,**kwargs)
axes[2].set_title("Inverted by rand boundary")
axes[3].imshow(pml_inverted,**kwargs)
axes[3].set_title("Inverted by PML")
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)


grad_rand = np.load(f'./results_rand/gradvpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
grad_pml = np.load(f'./results_pml/gradvpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(2,1,figsize=(5,5))
vmin, vmax=np.percentile(grad_rand, [2,98])
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(grad_rand,**kwargs)
axes[0].set_title("Gradient by random boundary")
axes[1].imshow(grad_pml,**kwargs)
axes[1].set_title("Gradient by PML")
plt.tight_layout()
plt.savefig("Gradient.png",dpi=300)
plt.show()

trace_no = 200
fig, ax=plt.subplots(1,1,figsize=(8,4))
ax.plot(true[:,trace_no],label="True")
ax.plot(init[:,trace_no],label="Initial")
ax.plot(rand_inverted[:,trace_no],label="Inverted by rand boundary")
ax.plot(pml_inverted[:,trace_no],label="Inverted by PML")
ax.legend()
ax.set_title("Trace")
plt.tight_layout()
plt.savefig("Trace.png",dpi=300)
plt.show()
