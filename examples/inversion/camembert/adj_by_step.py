import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

shot_no = 5
obs = np.load("./observed.npy", allow_pickle=True)[shot_no].squeeze()
ini = np.load("./initial.npy", allow_pickle=True)[shot_no].squeeze()
dx = 10
dt = 0.001
extent = [0, obs.shape[1]*dx, obs.shape[0]*dt, 0]

fig, axes = plt.subplots(1, 2, figsize=(4, 3))
axes[0].imshow(obs, extent=extent, cmap="seismic", aspect="auto")
axes[0].set_title("Observed")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("t (s)")
axes[1].imshow(ini, extent=extent, cmap="seismic", aspect="auto")
axes[1].set_title("Initial")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("t (s)")
plt.tight_layout()
plt.savefig("Observed.png", dpi=300)

# nim
square_obs = obs**2
square_ini = ini**2

intergal_obs = cumulative_trapezoid(square_obs, axis=0)
intergal_ini = cumulative_trapezoid(square_ini, axis=0)

intergal_obs/=np.max(intergal_obs)
intergal_ini/=np.max(intergal_ini)

fig, axes = plt.subplots(1, 2, figsize=(4, 3))

axes[0].imshow(intergal_obs, extent=extent, cmap="seismic", aspect="auto")
axes[0].set_title("Squared Observed")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("t (s)")
axes[1].imshow(intergal_ini, extent=extent, cmap="seismic", aspect="auto")
axes[1].set_title("Squared Initial")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("t (s)")
plt.tight_layout()
plt.savefig("Squared.png", dpi=300)

fig, ax=plt.subplots(1,1,figsize=(4,3))
ax.plot(intergal_obs[:, 100], label="Observed")
ax.plot(intergal_ini[:, 100], label="Initial")
ax.set_xlabel("t (ms)")
ax.set_ylabel("Cumulative Energy")
ax.legend()
plt.tight_layout()
