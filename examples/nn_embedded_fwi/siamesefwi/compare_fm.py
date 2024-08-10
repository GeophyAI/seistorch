import torch
from configure import seed, dt
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(seed)

from utils import SiameseCNN, show_freq_spectrum

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
cnn_before = SiameseCNN().to(dev)
cnn_after = torch.load('siamese.pth').to(dev)

# Load observed data
obs = np.load('obs_filtered.npy')
obs = torch.from_numpy(obs).float().unsqueeze(1).to(dev)

# Forward
latent_before = cnn_before(obs)
latent_after = cnn_after(obs)

# Compare
shot_no = 64
fig, axes=plt.subplots(1, 3, figsize=(10, 5))
latent_before = latent_before.detach().cpu().numpy().squeeze()[shot_no]
latent_after = latent_after.detach().cpu().numpy().squeeze()[shot_no]
vmin, vmax=np.percentile(latent_before, [1, 99])
kwargs=dict(vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
axes[0].imshow(obs[shot_no].detach().cpu().numpy().squeeze(), **kwargs)
axes[0].set_title('Observed')
axes[1].imshow(latent_before, **kwargs)
axes[1].set_title('Latent Before')
axes[2].imshow(latent_after, **kwargs)
axes[2].set_title('Latent After')
plt.tight_layout()
plt.show()

obs = obs[shot_no].detach().cpu().numpy().squeeze()
show_freq_spectrum(np.expand_dims(obs, 2), dt=dt, title='Frequency Spectrum of Observed data')
show_freq_spectrum(np.expand_dims(latent_before, 2), dt=dt*2, title='Frequency Spectrum of Latent Before')
show_freq_spectrum(np.expand_dims(latent_after, 2), dt=dt*2, title='Frequency Spectrum of Latent After')



