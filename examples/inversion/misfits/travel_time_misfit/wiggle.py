import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../../..")
from seistorch.show import SeisShow

show = SeisShow()

Observed = np.load("./observed.npy", allow_pickle=True)
Initial = np.load("./observed_init.npy", allow_pickle=True)
Inverted_l2 = np.load("./observed_invt_l2.npy", allow_pickle=True)
Inverted_tt = np.load("./observed_invt_tt.npy", allow_pickle=True)

shot_no = 20

show.wiggle([Observed[shot_no], Initial[shot_no], Inverted_l2[shot_no], Inverted_tt[shot_no]],
            ["red", "black", "green", "blue"], 
            ["Observed", "Initial", "L2", "TravelTime"], 
            dt=0.001, 
            dx=20, 
            downsample=20, 
            savepath="./wiggle.png")