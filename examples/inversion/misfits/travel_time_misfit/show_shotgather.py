import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader
np.random.seed(20230915)

import sys
sys.path.append("/home/shaowinw/seistorch")
from seistorch.show import SeisShow
show = SeisShow()
"""
Configures
"""
config_path = "./config/forward_obs.yml"
obsPath = "./observed.npy"
iniPath = "./observed_init.npy"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data
obs = np.load(obsPath, allow_pickle=True)
ini = np.load(iniPath, allow_pickle=True)

nshots = obs.shape[0]
nsamples, ntraces, ncomponent = obs[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

# show 5 shots randomly
showshots = np.random.randint(0, nshots, 5)

show.wiggle([obs[showshots[0]], ini[showshots[0]]],
            ["r", "b"],
            ["Observed","Initial"],
            dt=cfg['geom']['dt'],
            dx=cfg['geom']['h'],
            savepath="./wiggle.png", 
            downsample=20)