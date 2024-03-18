import numpy as np
import matplotlib.pyplot as plt

vel = np.load("./vel_rand.npy")
plt.imshow(vel, cmap="seismic")
plt.show()

cog_pml = np.load("shot_gather_pml.npy", allow_pickle=True)[0]
cog_rand = np.load("shot_gather_rand.npy", allow_pickle=True)[0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
kwargs={"cmap": "seismic", "aspect": "auto"}
axes[0].imshow(cog_pml, **kwargs)
axes[1].imshow(cog_rand, **kwargs)
plt.show()

timestep=2400
wfpml = np.load(f"./wf_pml/wf{timestep:04d}.npy")
wfrand = np.load(f"./wf_rand/wf{timestep:04d}.npy")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
vmin, vmax=np.percentile(wfpml[0, :, :], [2, 98])
kwargs = {"cmap": "gray", "vmin": vmin, "vmax": vmax, "aspect": "auto"}
axes[0].imshow(wfpml[0, :, :], **kwargs)
axes[1].imshow(wfrand[0, :, :], **kwargs)
plt.show()