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
center_vp = (nz//2, nx//2) # Center of the anomaly

vp = generate_anomaly_model(nx, nz, radius, center_vp, background_vp, anaomaly_vp)

fig, ax=plt.subplots(1,1,figsize=(5,3))
plt.colorbar(ax.imshow(vp, cmap="seismic", vmin=vp.min(), vmax=vp.max(), aspect="auto"), orientation='vertical')
ax.set_title("Vp")
ax.set_xlabel("x(Grid)")
ax.set_ylabel("z(Grid)")
plt.tight_layout()
plt.savefig("model.png")
plt.show()

# Save the model
os.makedirs("models", exist_ok=True)
np.save("models/vp.npy", vp.astype(np.float32))

print('Std of vp:', vp.std(), 'Mean of vp:', vp.mean())
