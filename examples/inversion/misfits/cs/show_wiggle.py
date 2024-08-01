import sys
import numpy as np
sys.path.append('../../')
from seistorch.show import SeisShow
import matplotlib.pyplot as plt

show = SeisShow()

obs = np.load("observed.npy", allow_pickle=True)
cs = np.load("observed_cs.npy", allow_pickle=True)
l2 = np.load("observed_l2.npy", allow_pickle=True)

shot_no = 30
# show.wiggle([obs[shot_no], cs[shot_no], l2[shot_no]], 
#             ["r", "b", "g"],
#             ["Observed", "CS", "L2"], 
#             downsample=10,
#             dx=20)
show.shotgather([obs[shot_no], cs[shot_no], l2[shot_no]],
                ["Observed", "CS", "L2"],
                dx=20,
                normalize=False,
                inarow=True)

fig, ax = plt.subplots(1,1,figsize=(10,6))
plt.plot(obs[shot_no][:, 0]/obs[shot_no][:, 0].max(), label="Observed")
plt.plot(cs[shot_no][:, 0]/cs[shot_no][:, 0].max(), label="CS")
plt.plot(l2[shot_no][:, 0]/l2[shot_no][:, 0].max(), label="L2")
plt.legend()
plt.show()