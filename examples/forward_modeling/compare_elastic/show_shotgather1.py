import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader
 
"""
Configures
"""
elasticPath = "./shot_gather_elastic.npy"
# Load the modeled data
obsElasticBG = np.load(elasticPath, allow_pickle=True)

nshots = obsElasticBG.shape[0]
nsamples, ntraces, ncomponent = obsElasticBG[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

# Plot the data
shot = 0
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
# show elastic data
start = 0
end = -1
startt = 0
show_data = [obsElasticBG[shot][startt:, start:end, 0]]
ax.imshow(show_data[0], cmap="seismic", aspect="auto")
ax.set_xlabel("Trace")
ax.set_ylabel("Time sample")

plt.tight_layout()
# plt.savefig("shot_gather.png", dpi=300)
plt.show()