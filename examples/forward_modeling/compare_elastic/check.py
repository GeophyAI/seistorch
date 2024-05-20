import numpy as np
import glob
import matplotlib.pyplot as plt

wffiles = sorted(glob.glob("wf_pml/*.npy"))
pmln = 50

for i, wffile in enumerate(wffiles):
    if i %200==0:
        wf = np.load(wffile)[:, pmln:-pmln, pmln:-pmln]
        print(wf.shape)
        vmin,vmax=np.percentile(wf[0], [2, 98])
        plt.imshow(wf[0],  vmin=vmin, vmax=vmax, 
                #    interpolation="nearest",
                   cmap="gray")
        plt.title(f"i={i}")
        plt.show()