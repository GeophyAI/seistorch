import glob
import numpy as np
import matplotlib.pyplot as plt

pmln = 50
wf_files = sorted(glob.glob("./wavefield/*.npy"))

for i, wf_file in enumerate(wf_files):
    wf = np.load(wf_file)[0][0]
    wf = wf[pmln:-pmln, pmln:-pmln]
    if i %100==0:
        plt.imshow(wf)
        plt.show()

