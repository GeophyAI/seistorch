import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
time = "0700"
vti_a = np.load(f"./vti_a/wf_foward{time}.npy")[0]
vti_b = np.load(f"./vti_b/wf_foward{time}.npy")[0]
vti_c = np.load(f"./vti_c/wf_foward{time}.npy")[0]
titles = [r"$\delta= \epsilon$=0.3", 
          r"$\delta=0.1, \epsilon=0.3$",
          r"$\delta=0.3, \epsilon=0.1$"]
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for d, ax, title in zip([vti_a, vti_b, vti_c], ax, titles):
    vmin, vmax=np.percentile(d, [1, 99])
    ax.imshow(d, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=18)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"figures/wavefield_{time}.png", dpi=300)
plt.show()
