import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 0
epoch = 49
true = np.load('./velocity_model/true.npy')
init = np.load('./velocity_model/init.npy')
inverted_l2 = np.load(f'./results/l2/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
inverted_tt = np.load(f'./results/traveltime/paravpF00E{epoch:02d}.npy')[npml:-npml, npml+expand:-npml-expand]


fig, ax=plt.subplots(1,1,figsize=(5,3))
vmin,vmax=true.min(),true.max()
trace = 700
ax.plot(np.arange(true.shape[0]), true[:,trace], 'r',label='True')
ax.plot(np.arange(true.shape[0]), init[:,trace], 'b',label='Initial')
ax.plot(np.arange(true.shape[0]), inverted_l2[:,trace], 'g',label='L2')
ax.plot(np.arange(true.shape[0]), inverted_tt[:,trace], 'black',label='Traveltime')

plt.legend()
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)
np.save(f"./velocity_model/inverted_l2.npy", inverted_l2)
np.save(f"./velocity_model/inverted_tt.npy", inverted_tt)
