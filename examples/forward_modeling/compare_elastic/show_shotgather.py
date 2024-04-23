import numpy as np
import matplotlib.pyplot as plt

from yaml import load
from yaml import CLoader as Loader

"""
Configures
"""
elasticPath = "./shot_gather_elastic.npy"
vdrPath = "./shot_gather_vdr.npy"
elasticBGPath = "./shot_gather_elastic_bg.npy"
vdrBGPath = "./shot_gather_vdr_bg.npy"
# Load the modeled data
obsElasticBG = np.load(elasticBGPath, allow_pickle=True)
obsVdrBG = np.load(vdrBGPath, allow_pickle=True)
# Load the Direct data
obsElastic = np.load(elasticPath, allow_pickle=True)
obsVdr = np.load(vdrPath, allow_pickle=True)

nshots = obsElastic.shape[0]
nsamples, ntraces, ncomponent = obsElastic[0].shape

print(f"The data has {nshots} shots, {nsamples} time samples, {ntraces} traces, and {ncomponent} components.")

# Plot the data
shot = 0
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
# show elastic data
start = 64
end = 192
startt = 800
show_data = [obsElastic[shot][startt:, start:end, 0]-obsElasticBG[shot][startt:, start:end, 0],
             obsElastic[shot][startt:, start:end, 1]-obsElasticBG[shot][startt:, start:end, 1],
             obsVdr[shot][startt:, start:end, 0]-obsVdrBG[shot][startt:, start:end, 0],
             obsVdr[shot][startt:, start:end, 1]-obsVdrBG[shot][startt:, start:end, 1]]
for i, ax in enumerate(axes.ravel()):
    ax.imshow(show_data[i], cmap="seismic", aspect="auto")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time sample")

plt.tight_layout()
# plt.savefig("shot_gather.png", dpi=300)
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))
# compare vx
trace = 50
e_trace = obsElastic[shot][:, trace, 0]-obsElasticBG[shot][:, trace, 0]
v_trace = obsVdr[shot][:, trace, 0]-obsVdrBG[shot][:, trace, 0]
axes[0].plot(e_trace/e_trace.max(), label="elastic")
axes[0].plot(v_trace/v_trace.max(), label="vdr")
axes[0].set_xlabel("Time sample")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
# compare vz
e_trace = obsElastic[shot][:, trace, 1]-obsElasticBG[shot][:, trace, 1]
v_trace = obsVdr[shot][:, trace, 1]-obsVdrBG[shot][:, trace, 1]
axes[1].plot(e_trace/e_trace.max(), label="elastic")
axes[1].plot(v_trace/v_trace.max(), label="vdr")
axes[1].set_xlabel("Time sample")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
plt.show()