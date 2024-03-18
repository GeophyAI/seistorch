import numpy as np
import glob
import matplotlib.pyplot as plt

# pmlc = np.load("pml.npy")

# plt.imshow(pmlc)
# plt.show()

# plt.plot(pmlc[:,100])
# plt.show()

# tm = np.load("tm.npy")
# bm = np.load("bm.npy")
# lm = np.load("lm.npy")
# rm = np.load("rm.npy")
# for m in [tm, bm, lm, rm]:
#     plt.imshow(m)
#     plt.show()

wffiles = sorted(glob.glob("wf_pml/*.npy"))
pmln = 50

for i, wffile in enumerate(wffiles):
    if i %200==0 and i <2500:
        wf = np.load(wffile)#[:, pmln:-pmln, pmln:-pmln]
        #vmin,vmax=np.percentile(wf[0], [1, 99]) vmin=vmin, vmax=vmax, 
        plt.imshow(wf[0], 
                   interpolation="nearest",
                   cmap="seismic")
        plt.title(f"i={i}")
        plt.colorbar()
        plt.show()