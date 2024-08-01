import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import convolve2d as conv2d

class SiameseCNN(nn.Module):
    def __init__(self, init='kaiming'):
        super(SiameseCNN, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        ])
        # initialize weights
        if init == 'kaiming':
            for conv in self.conv_layers:
                nn.init.kaiming_normal_(conv.weight.data)
        elif init == 'xavier':
            for conv in self.conv_layers:
                nn.init.xavier_normal_(conv.weight.data)
        elif init == 'normal':
            for conv in self.conv_layers:
                nn.init.normal_(conv.weight.data)
        else:
            raise NotImplementedError

    def forward(self, x):
        for idx in range(len(self.conv_layers)):
            x = self.conv_layers[idx](x)
            x = F.relu(x)
        return x
    
siamese_kaiming = SiameseCNN(init='kaiming')
siamese_xavier = SiameseCNN(init='xavier')
siamese_normal = SiameseCNN(init='normal')
# Load data
obs = np.load('obs.npy')
obs = torch.from_numpy(obs).float().unsqueeze(1)
with torch.no_grad():
    latent_obs_kaiming = siamese_kaiming(obs)
    latent_obs_xavier = siamese_xavier(obs)
    latent_obs_normal = siamese_normal(obs)

# show data
show_number = 1
rand_no = np.random.randint(0, len(obs), show_number)
for no in range(show_number):
    shot_no = rand_no[no]
    fig, axes = plt.subplots(1, 4, figsize=(8, 3))
    _obs = obs[shot_no].cpu().numpy().squeeze()
    vmin, vmax = np.percentile(_obs, [1, 99])
    axes[0].imshow(_obs, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
    axes[0].set_title('Observed')
    axes[0].axis('off')
    # kaiming
    latent_obs = latent_obs_kaiming[shot_no].cpu().detach().numpy().squeeze()
    vmin, vmax = np.percentile(latent_obs, [1, 99])
    axes[1].imshow(latent_obs, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
    axes[1].set_title('Kaiming')
    axes[1].axis('off')
    # xavier
    latent_obs = latent_obs_xavier[shot_no].cpu().detach().numpy().squeeze()
    vmin, vmax = np.percentile(latent_obs, [1, 99])
    axes[2].imshow(latent_obs, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
    axes[2].set_title('Xavier')
    axes[2].axis('off')
    # normal
    latent_obs = latent_obs_normal[shot_no].cpu().detach().numpy().squeeze()
    vmin, vmax = np.percentile(latent_obs, [1, 99])
    axes[3].imshow(latent_obs, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
    axes[3].set_title('Normal')
    axes[3].axis('off')
    plt.tight_layout()
plt.show()

# normal_kernel = siamese_normal.conv_layers[0].weight.data.numpy().squeeze()
# conv = conv2d(normal_kernel, obs[0].numpy().squeeze())
# vmin, vmax = np.percentile(conv, [1, 99])
# plt.imshow(conv, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
# plt.title('Normal Convolution')
# plt.show()

