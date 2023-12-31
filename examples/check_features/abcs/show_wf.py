import numpy as np
import matplotlib.pyplot as plt

visa = np.load("./wavefield_visa/wf0800.npy")[0][50:-50, 50:-50]
acoustic = np.load("./wavefield_acoustic/wf0800.npy")[0][50:-50, 50:-50]
elastic = np.load("./wavefield_elastic/wf0800.npy")[0][50:-50, 50:-50]
tti = np.load("./wavefield_tti/wf0800.npy")[0][50:-50, 50:-50]

# wf = np.load("./wavefield/wf0800.npy")[0][50:-50, 50:-50]
# wf[np.where(np.abs(wf)<1e-3)] = 0
vmin,vmax=np.percentile(tti, [2,97])

fig,ax=plt.subplots(figsize=(6,6))
plt.imshow(visa, vmin=vmin, vmax=vmax, cmap="seismic")
plt.show()
fig.savefig("visa.png", dpi=300, bbox_inches="tight")