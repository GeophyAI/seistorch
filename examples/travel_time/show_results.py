import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 0
epoch = 49
true = np.load('./velocity_model/true.npy')
init = np.load('./velocity_model/init.npy')
inverted = np.load(f'./results/traveltime/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
grad = np.load(f'./results/traveltime/gradvpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
fig, ax=plt.subplots(1,1,figsize=(5,3))
vmin,vmax=true.min(),true.max()
trace = 700
ax.plot(np.arange(true.shape[0]), true[:,trace], 'r',label='True')
ax.plot(np.arange(true.shape[0]), init[:,trace], 'b',label='Initial')
ax.plot(np.arange(true.shape[0]), inverted[:,trace], 'g',label='Inverted')
plt.legend()
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)
np.save("./velocity_model/inverted.npy", inverted)
fig, ax=plt.subplots(1,1,figsize=(5,3))
vmin, vmax=np.percentile(grad, [2,98])
ax.imshow(grad, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)
plt.tight_layout()
plt.colorbar(ax.images[0], ax=ax)
plt.show()
