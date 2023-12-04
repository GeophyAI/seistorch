import numpy as np
import matplotlib.pyplot as plt

nt = 1499
ct = 1200

wfb = np.load(f"./backward/backward{ct:04d}.npy")[:,50:-50, 50:-50]
wff = np.load(f"./forward/forward{nt-ct:04d}.npy")[:,50:-50, 50:-50]

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(wfb[0].squeeze(), cmap='gray', aspect='auto')
axes[1].imshow(wff[0].squeeze(), cmap='gray', aspect='auto')
axes[2].imshow(wfb[0].squeeze()-wff[0].squeeze(), cmap='gray', aspect='auto')
plt.show()