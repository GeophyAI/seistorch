import torch
import numpy as np
import matplotlib.pyplot as plt


siamese = torch.load("siamese.pth")
obs = np.load("obs.npy")
obs = torch.from_numpy(obs).float().unsqueeze(1).cuda()

latent_obs = siamese(obs)
latent_obs = latent_obs.cpu().detach().numpy().squeeze()

# show sameple
show_number = 5
rand_no = np.random.randint(0, len(obs), show_number)
for i in range(show_number):
    plt.figure(figsize=(5, 3))
    _obs = obs[rand_no[i]].cpu().numpy().squeeze()
    vmin, vmax = np.percentile(_obs, [1, 99])
    plt.imshow(_obs, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    plt.title("Observed")
    plt.show()

    fm_counts = latent_obs.shape[1]
    cols = rows = np.sqrt(fm_counts) 
    assert cols == int(cols)
    cols = int(cols)
    rows = int(rows)
    fit, axes = plt.subplots(cols, rows, figsize=(10, 8))
    for j, ax in enumerate(axes.ravel()):
        vmin, vmax = np.percentile(latent_obs[rand_no[i]][j], [1, 99])
        ax.imshow(latent_obs[rand_no[i]][j], vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.set_title(f"Feature map {j}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
