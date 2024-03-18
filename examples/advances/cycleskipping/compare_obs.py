import numpy as np
import sys
sys.path.append("../..")

from seistorch.show import SeisShow

show = SeisShow()

obsvered_true = np.load("./observed.npy", allow_pickle=True)
obsvered_init = np.load("./observed_init.npy", allow_pickle=True)

shot_no = 30

show.shotgather([obsvered_true[shot_no], obsvered_init[shot_no]], 
                ["Observed", "Initial"], 
                inarow=True,
                dt=0.001, 
                aspect="auto",
                dx=12.5)

show.wiggle([obsvered_true[shot_no], obsvered_init[shot_no]], 
            ["red", "black"], 
            ["Observed", "Initial"], 
            dt=0.001, 
            dx=12.5, 
            downsample=10)