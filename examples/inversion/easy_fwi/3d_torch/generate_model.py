import numpy as np
import matplotlib.pyplot as plt
from configures import nz, ny, nx
from scipy.ndimage import gaussian_filter1d
import os

model = np.ones((nz, ny, nx), dtype=np.float32) * 1500.
model[30:, :, :] = 2000.

#    ________
#  /   .    /|
# /________/ |
# | 1500m/s| |
# |________|/|
# | 2000m/s| |
# |________|/
#

# smooth along the y-axis
sm_model = model.copy()
sm_model = gaussian_filter1d(sm_model, sigma=11, axis=0)

# plot the model
plt.figure(figsize=(5, 3))
plt.plot(sm_model[:, 50, 50], label="Smoothed model")
plt.plot(model[:, 50, 50], label="Original model")
plt.legend()
plt.title("Y-sclie")
plt.show()

os.makedirs("models", exist_ok=True)
np.save("models/true_vp.npy", model)
np.save("models/init_vp.npy", sm_model)
