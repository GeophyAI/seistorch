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

wffiles = sorted(glob.glob("wavefield_acoustic/*.npy"))
pmln = 30

for i, wffile in enumerate(wffiles):
    if i %50==0 and i <1500:
        wf = np.load(wffile)#[:, pmln:-pmln, pmln:-pmln]
        #vmin,vmax=np.percentile(wf[0], [1, 99]) vmin=vmin, vmax=vmax, 
        plt.imshow(wf[0], cmap="seismic")
        plt.show()