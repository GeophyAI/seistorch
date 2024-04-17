import numpy as np
import matplotlib.pyplot as plt

timestep = 600
bwidth = 50
acoustic = np.load(f"./wf_acoustic/wf_foward{timestep:04d}.npy")[0][bwidth:-bwidth, bwidth:-bwidth]
vti = np.load(f"./wf_vti/wf_foward{timestep:04d}.npy")[0][bwidth:-bwidth, bwidth:-bwidth]
vacoustic_all = np.load(f"./wf_vacoustic/wf_foward{timestep:04d}.npy")[0][bwidth:-bwidth, bwidth:-bwidth]
vacoustic_aonly = np.load(f"./wf_vacoustic_aonly/wf_foward{timestep:04d}.npy")[0][bwidth:-bwidth, bwidth:-bwidth]
vacoustic_ponly = np.load(f"./wf_vacoustic_ponly/wf_foward{timestep:04d}.npy")[0][bwidth:-bwidth, bwidth:-bwidth]

fig, axes = plt.subplots(1,2, figsize=(6,3))
axes[0].imshow(acoustic, cmap="seismic")
axes[0].set_title("Acoustic")
axes[1].imshow(vti, cmap="seismic")
axes[1].set_title("VTI")
for ax in axes.flatten():
    ax.axis("off")
plt.tight_layout()
plt.show()
fig.savefig("acoustic_vti.png", dpi=300, bbox_inches="tight")

# combine acoustic and vacoustic

combine = np.zeros_like(acoustic)
center = acoustic.shape[1]//2
# top left
combine[:center, :center] = acoustic[:center, :center]
# top right
combine[:center, center:] = vacoustic_aonly[:center, center:]
# bottom left
combine[center:, :center] = vacoustic_ponly[center:, :center]
# bottom right
combine[center:, center:] = vacoustic_all[center:, center:]
fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(combine, cmap="seismic", aspect='auto')
ax.text(56, 120, "Acoustic", color="white", fontsize=12)
ax.text(130, 120, "Amplitude", color="white", fontsize=12)
ax.text(56, 145, "Phase", color="white", fontsize=12)
ax.text(130, 145, "All-decay", color="white", fontsize=12)
ax.axis('off')
ax.vlines(128, 0, 256, color="black", linestyle="--")
ax.hlines(128, 0, 256, color="black", linestyle="--")
plt.tight_layout()
plt.show()
fig.savefig("visco_acoustic.png", dpi=300, bbox_inches="tight")

