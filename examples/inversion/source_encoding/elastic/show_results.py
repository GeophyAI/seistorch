import numpy as np
import matplotlib.pyplot as plt
import torch

npml = 50
expand = 50
true_vp = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
true_vs = np.load('../../../models/marmousi_model/true_vs.npy')[:,expand:-expand]

init_vp = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
init_vs = np.load('../../../models/marmousi_model/linear_vs.npy')[:,expand:-expand]

inverted = torch.load('./results/model_F00E49.pt')
inverted_vp = inverted['vp'].detach().cpu().numpy()[npml:-npml, npml+expand:-npml-expand]
inverted_vs = inverted['vs'].detach().cpu().numpy()[npml:-npml, npml+expand:-npml-expand]

nz, nx = true_vp.shape
vmin_vp, vmax_vp = true_vp.min(), true_vp.max()
vmin_vs, vmax_vs = true_vs.min(), true_vs.max()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))
titles = ["True vp", "True vs", "Init vp", "Init vs", "Inverted vp", "Inverted vs"]
for d, ax, title in zip([true_vp, true_vs,
                        init_vp, init_vs, 
                        inverted_vp, inverted_vs], axes.ravel(), titles):
    if "vp" in title:
        vmin, vmax = vmin_vp, vmax_vp
    else:
        vmin, vmax = vmin_vs, vmax_vs
    kwargs = dict(cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax, extent=[0, nx, nz, 0])
    _ax_ = ax.imshow(d, **kwargs)
    plt.colorbar(_ax_, ax=ax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("Inverted.png", dpi=300)
plt.show()


