import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")
from seistorch.show import SeisShow
show = SeisShow()
npml = 50
expand = 50
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
inverted_withlow = np.load('./results/towed_withlow/paravpF02E49.npy')[npml:-npml, npml+expand:-npml-expand]
inverted_withoutlow = np.load('./results/towed_withoutlow/paravpF01E49.npy')[npml:-npml, npml+expand:-npml-expand]
nz,nx = true.shape

fig, axes=plt.subplots(5,1,figsize=(8,15))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax, "extent":[0,nx*20,nz*20,0]}
axes[0].imshow(init,**kwargs)
axes[0].set_title("Initial")
axes[1].imshow(inverted_withlow,**kwargs)
axes[1].set_title("Inverted with low")
axes[2].imshow(inverted_withoutlow,**kwargs)
axes[2].set_title("Inverted without low")

trace = 200
axes[3].plot(true[:,trace], color="black", label="True")
axes[3].plot(init[:,trace], color="red", label="Initial")
axes[3].plot(inverted_withlow[:,trace], color="blue", label="Inverted with low frequency")
axes[3].plot(inverted_withoutlow[:,trace], color="green", label="Inverted without low frequency")
axes[3].legend()
axes[3].set_title(f"Trace 4km")

trace = 300
axes[4].plot(true[:,trace], color="black", label="True")
axes[4].plot(init[:,trace], color="red", label="Initial")
axes[4].plot(inverted_withlow[:,trace], color="blue", label="Inverted with low frequency")
axes[4].plot(inverted_withoutlow[:,trace], color="green", label="Inverted without low frequency")
axes[4].legend()
axes[4].set_title(f"Trace 6km")

plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)

