import numpy as np
import matplotlib.pyplot as plt

obs = np.load("./results/obs.npy")
syn = np.load("./results/syn.npy")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(obs.squeeze(), cmap="gray", aspect="auto")
axes[0].set_title("Observed data")
axes[1].imshow(syn.squeeze(), cmap="gray", aspect="auto")
axes[1].set_title("Synthetic data")
plt.show()
