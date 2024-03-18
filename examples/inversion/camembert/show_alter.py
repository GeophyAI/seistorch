import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader

import sys
sys.path.append("../../..")
from seistorch.show import SeisShow

show = SeisShow()

"""
Configures
"""
config_path = "./observed.yml"
obsPath = "./observed.npy"
iniPath = "./initial.npy"

loss ="l2"
obsPath = f"./results_{loss}/obs.npy"
iniPath = f"./results_{loss}/syn.npy"
# adjPath = f"./results_{loss}/adj.npy"

# Load the configure file
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
# Load the modeled data
obs = np.load(obsPath, allow_pickle=True)
ini = np.load(iniPath, allow_pickle=True)
# adj = np.load(adjPath, allow_pickle=True)

nshots = obs.shape[0]
nums_show = 8
nsamples, ntraces, ncomponent = obs[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

shot_no = 0
obs = obs[shot_no].squeeze()
ini = ini[shot_no].squeeze()
# adj = adj[shot_no].squeeze()

show.shotgather([obs, ini], 
                ["Observed", "Initial"], 
                dt=cfg['geom']['dt'], 
                normalize=False,
                inarow=True,
                dx=cfg['geom']['h'],
                savepath="obs_ini.png")

show.alternate(obs, 
               ini,
               interval=20,
               cmap=plt.cm.seismic,
               trace_normalize=False,
               dt=cfg['geom']['dt'], 
               dx=cfg['geom']['h'])