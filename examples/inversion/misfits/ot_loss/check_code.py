import numpy as np
import matplotlib.pyplot as plt

obs = np.load("./results_l2/obs.npy", allow_pickle=True)
syn = np.load("./results_l2/syn.npy", allow_pickle=True)

for i in range(obs.shape[0]):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(obs[i], cmap="seismic", aspect="auto")
    axes[0].set_title("Observed")
    axes[1].imshow(syn[i], cmap="seismic", aspect="auto")
    axes[1].set_title("Synthetic")
    plt.tight_layout()
    plt.show()
