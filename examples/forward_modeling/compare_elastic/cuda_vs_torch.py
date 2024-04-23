import numpy as np
import matplotlib.pyplot as plt

nx, nz = 600, 400
bwidth = 50
order = 1
pad = bwidth + order
time = 1200
cuda = np.fromfile(f"/home/shaowinw/synthetic/habc_aec/Gif/P{time:04d}.dat", dtype=np.float32)
cuda = cuda[5:].reshape(nz+pad*2, nx+pad*2)
cuda = cuda[pad:-pad,pad:-pad]

torch = np.load(f"./wf_pml/wf_foward{time-0:04d}.npy")[0][50:-50, 50:-50]

assert cuda.shape == torch.shape, "The shape of the wavefield is not the same."

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(cuda, cmap="gray")
axes[0].set_title("CUDA")
axes[1].imshow(torch, cmap="gray")
axes[1].set_title("Torch")
axes[2].imshow(torch-cuda, cmap="gray")
axes[2].set_title("Torch")
plt.tight_layout()
plt.show()

