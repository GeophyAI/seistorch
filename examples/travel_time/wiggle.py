import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")
from seistorch.show import SeisShow

show = SeisShow()

Observed = np.load("./observed.npy", allow_pickle=True)
Initial = np.load("./observed_init.npy", allow_pickle=True)
Inverted = np.load("./observed_invt.npy", allow_pickle=True)

shot_no = 50

show.wiggle([Observed[shot_no], Initial[shot_no], Inverted[shot_no]],
            ["red", "black", "green"], 
            ["Observed", "Initial", "Inverted"], 
            dt=0.001, 
            dx=20, 
            downsample=20, 
            savepath="./wiggle.png")

