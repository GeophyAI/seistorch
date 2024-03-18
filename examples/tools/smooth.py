import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

true = np.load("../models/marmousi_model_half/true_vp.npy")

for sigma in range(10):
    init = true.copy()

    gaussian_filter(init, sigma=sigma, output=init)

    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    kwargs = dict(vmin=1500, vmax=5500, cmap="seismic", aspect="auto")
    axes.imshow(init, **kwargs)
    plt.tight_layout()
    plt.axis("off") 
    plt.show()
    fig.savefig(f"sigma={sigma}.png", dpi=300)