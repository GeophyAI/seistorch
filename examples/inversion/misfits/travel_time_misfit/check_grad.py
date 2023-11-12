import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 0
epoch = 1

grad = np.load(f"./results/ip/gradvpF00E{epoch:02d}.npy")
vmin, vmax = np.percentile(grad, [5, 95])
plt.imshow(grad, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
plt.colorbar()
plt.show()

true = np.load('./velocity_model/true.npy')
init = np.load('./velocity_model/init.npy')
inverted_ip = np.load(f'./results/ip/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

# Show inverted 
fig, ax=plt.subplots(1,1,figsize=(5,3))
vmin,vmax=true.min(),true.max()
trace = 400
ax.plot(np.arange(true.shape[0]), true[:,trace], 'r',label='True')
ax.plot(np.arange(true.shape[0]), init[:,trace], 'b',label='Initial')
ax.plot(np.arange(true.shape[0]), inverted_ip[:,trace], 'g',label='L2')

plt.legend()
plt.tight_layout()