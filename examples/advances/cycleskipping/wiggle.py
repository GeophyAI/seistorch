import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../..")
from seistorch.show import SeisShow
show = SeisShow()

ini = np.load("./observed_init.npy", allow_pickle=True)
obs = np.load("./observed.npy", allow_pickle=True)
inv = np.load("./observed_invt.npy", allow_pickle=True)

shot_no = 50

show.wiggle([ini[shot_no], obs[shot_no], inv[shot_no]], 
            ["red", "black", "blue"],
            ["Initial", "Observed", "Inverted"], 
            dt=0.001, 
            dx=20, 
            downsample=20)