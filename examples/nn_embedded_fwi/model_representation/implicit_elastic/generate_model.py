import numpy as np
import matplotlib.pyplot as plt
import os
from configure import *
os.makedirs("models", exist_ok=True)
# Generate a anormaly model, which is a circle with a different velocity
def generate_anomaly_model(nx, nz, radius, center, bg_velocity, circle_velocity):
    model = np.ones((nz, nx)) * bg_velocity
    for i in range(nz):
        for j in range(nx):
            if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                model[i, j] = circle_velocity
    return model

# Generate a simple model with a circle anomaly
center_vp = (nz//2, nx//4) # Center of the anomaly
center_vs = (nz//2, 3*nx//4) # Center of the anomaly
vp = generate_anomaly_model(nx, nz, radius, center_vp, 1500, 2500)
vs = generate_anomaly_model(nx, nz, radius, center_vs, 1500/vp_vs_ratio, 2500/vp_vs_ratio)
rho = 2000 * np.ones((nz, nx))

fig, axes=plt.subplots(1,2,figsize=(7,2))
plt.colorbar(axes[0].imshow(vp, cmap="seismic", vmin=1500, vmax=2500, aspect="auto"))
axes[0].set_title("Vp")
plt.colorbar(axes[1].imshow(vs, cmap="seismic", vmin=1500/vp_vs_ratio, vmax=2500/vp_vs_ratio, aspect="auto"))
axes[1].set_title("Vs")
plt.tight_layout()
plt.savefig("models/model.png")
plt.show()

# Save the model
os.makedirs("models", exist_ok=True)
np.save("models/vp.npy", vp.astype(np.float32))
np.save("models/vs.npy", vs.astype(np.float32))
np.save("models/rho.npy", rho.astype(np.float32))