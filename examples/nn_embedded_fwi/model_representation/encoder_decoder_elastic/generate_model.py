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

def generate_v_model(nx, nz, side_length, center, outside_velocity, inside_velocity):
    model = np.full((nz, nx), outside_velocity, dtype=float)
    cz, cx = center
    
    half_length = side_length / 2
    height = half_length 

    for ix in range(nx):
        for iz in range(nz):
            x = ix - cx
            z = iz - cz
            
            if (z >= 0) and (np.abs(x) <= (half_length - z * (half_length / height))):
                model[iz, ix] = inside_velocity

    return model

# Generate a simple model with a circle anomaly
center_vp = (nz//2, nx//4) # Center of the anomaly
center_vs = (nz//2, 3*nx//4) # Center of the anomaly
center_rho = (nz//2, 2*nx//4) # Center of the anomaly

vp = generate_anomaly_model(nx, nz, radius, center_vp, background_vp, anaomaly_vp)
vs = generate_anomaly_model(nx, nz, radius, center_vs, background_vp/vp_vs_ratio, anaomaly_vp/vp_vs_ratio)
rho = generate_v_model(nx, nz, 25, center_rho, background_rho, anaomaly_rho)
constant_rho = 2000 * np.ones((nz, nx))

bg_vp = np.ones((nz, nx)) * background_vp
bg_vs = np.ones((nz, nx)) * background_vp/vp_vs_ratio
bg_rho = np.ones((nz, nx)) * background_rho

fig, axes=plt.subplots(1,3,figsize=(10,3))
plt.colorbar(axes[0].imshow(vp, cmap="seismic", vmin=vp.min(), vmax=vp.max(), aspect="auto"), orientation='horizontal')
axes[0].set_title("Vp")
plt.colorbar(axes[1].imshow(rho, cmap="seismic", vmin=rho.min(), vmax=rho.max(), aspect="auto"), orientation='horizontal')
axes[1].set_title("Rho")
plt.colorbar(axes[2].imshow(vs, cmap="seismic", vmin=vs.min(), vmax=vs.max(), aspect="auto"), orientation='horizontal')
axes[2].set_title("Vs")
plt.tight_layout()
plt.savefig("models/model.png")
plt.show()

# Save the model
os.makedirs("models", exist_ok=True)
np.save("models/vp.npy", vp.astype(np.float32))
np.save("models/vs.npy", vs.astype(np.float32))
np.save("models/rho.npy", rho.astype(np.float32))
np.save("models/constant_rho.npy", constant_rho.astype(np.float32))

np.save("models/bg_vp.npy", bg_vp.astype(np.float32))
np.save("models/bg_vs.npy", bg_vs.astype(np.float32))
np.save("models/bg_rho.npy", bg_rho.astype(np.float32))

print('Std of vp:', vp.std(), 'Mean of vp:', vp.mean())
print('Std of vs:', vs.std(), 'Mean of vs:', vs.mean())
print('Std of rho:', rho.std(), 'Mean of rho:', rho.mean())