import numpy as np
import matplotlib.pyplot as plt

loss_tt = np.load("./results/traveltime/loss.npy")[0]
loss_l2 = np.load("./results/l2/loss.npy")[0]

loss_tt = np.sum(loss_tt, axis=-1)
loss_l2 = np.sum(loss_l2, axis=-1)

plt.plot(loss_tt/loss_tt.max(), label="Traveltime")
plt.plot(loss_l2/loss_l2.max(), label="L2")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

npml = 50
expand = 0
epoch = 30
true = np.load('./velocity_model/true.npy')
init = np.load('./velocity_model/init.npy')
inverted_l2 = np.load(f'./results/l2/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
inverted_tt = np.load(f'./results/traveltime/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
grad_tt = np.load(f'./results/traveltime/gradvpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
grad_l2 = np.load(f'./results/l2/gradvpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

# Show gradient
fig, axes=plt.subplots(1,2,figsize=(8,3))
vmin,vmax=np.percentile(grad_tt, [2, 98])
axes[0].imshow(grad_tt, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
axes[0].set_title("Traveltime")
vmin,vmax=np.percentile(grad_l2, [2, 98])
axes[1].set_title("L2")
axes[1].imshow(grad_l2, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
plt.tight_layout() 
plt.show()

# Show Velocity
fig, axes=plt.subplots(2,2,figsize=(6,4))
vmin,vmax=true.min(),true.max()
axes[0,0].imshow(true, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
axes[0,0].set_title("True")
axes[0,1].imshow(init, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
axes[0,1].set_title("Init")
axes[1,0].imshow(inverted_tt, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
axes[1,0].set_title("Traveltime")
axes[1,1].imshow(inverted_l2, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
axes[1,1].set_title("L2")
plt.tight_layout() 
plt.show()

# Show inverted 
fig, ax=plt.subplots(1,1,figsize=(5,3))
vmin,vmax=true.min(),true.max()
trace = 400
ax.plot(np.arange(true.shape[0]), true[:,trace], 'r',label='True')
ax.plot(np.arange(true.shape[0]), init[:,trace], 'b',label='Initial')
ax.plot(np.arange(true.shape[0]), inverted_l2[:,trace], 'g',label='L2')
ax.plot(np.arange(true.shape[0]), inverted_tt[:,trace], 'black',label='Traveltime')

plt.legend()
plt.tight_layout()
fig.savefig("Inverted.png",dpi=300)
np.save(f"./velocity_model/inverted_l2.npy", inverted_l2)
np.save(f"./velocity_model/inverted_tt.npy", inverted_tt)
