import numpy as np
import matplotlib.pyplot as plt
import glob
wf_files = sorted(glob.glob('./wf_habc/*.npy'))
dt = 0.002
for idx, file in enumerate(wf_files):
    if idx %100==0:
        wf = np.load(file)

        plt.imshow(wf[5], aspect='auto')
        plt.title(f'time={idx*dt}s')
        plt.show()
