import numpy as np
import matplotlib.pyplot as plt

type = 'results_implicit_siren'
# type = 'results_implicit_sirenscale'

# type = 'results_implicit_np_encoder'
# type = 'results_implicit_np_cnn'

npml = 50
expand = 50
epoch = 5600
true = np.load('../../models/marmousi_model/true_vp.npy')[:,expand:-expand]/1000
init = np.load('../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]/1000
# inverted = np.load('./results_explicit/paravpF00E99.npy')[npml:-npml, npml+expand:-npml-expand]
inverted = np.load(f'./{type}/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
loss = np.load(f"./{type}/loss.npy")
plt.plot(np.log10(loss.flatten()))
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

trace = 200
fig, ax=plt.subplots(1,1,figsize=(6,4))
ax.plot(true[:,trace],label="True")
# ax.plot(init[:,trace],label="Initial")
ax.plot(inverted[:,trace],label="Inverted")
ax.legend()
fig.savefig("Inverted_profile.png",dpi=300)
plt.show()

np.save("ipinverted.npy", np.pad(inverted, ((0, 0), (npml, npml)), 'edge'))
print(inverted.max(), inverted.min())


i1 = np.load(f'./{type}/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
i2 = np.load(f'./{type}/paravpF00E{epoch-1:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

print((i1-i2).max(), (i1-i2).min())
