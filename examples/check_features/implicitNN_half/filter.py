import numpy as np
import matplotlib.pyplot as plt
import lesio

obs = np.load("observed.npy", allow_pickle=True)    
for low in [3.0, 5.0, 8.0, 10.0]:
    fd = lesio.tools.fitler_fft(obs[18], dt=0.002, N=4, low=low, axis=0, mode="lowpass")
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    kwargs = dict(vmin=-0.1, vmax=0.1, cmap="seismic", aspect="auto")
    axes.imshow(fd[:1500], **kwargs)
    plt.tight_layout()
    plt.axis("off") 
    plt.show()
    fig.savefig(f"low={low}.png", dpi=300, bbox_inches="tight")