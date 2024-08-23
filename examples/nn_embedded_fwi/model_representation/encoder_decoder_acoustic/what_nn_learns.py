import torch
import numpy as np
import matplotlib.pyplot as plt
from configure import *
from networks import FWINET

obs = np.load("obs.npy")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs = torch.from_numpy(obs).float().to(dev)
obs = obs.unsqueeze(0)

# load etmpy model
epoch = '500'
_, shots, nsamples, nrecs = obs.shape
net = FWINET(shots, nsamples, nrecs, min_filters=min_filters, latent_length=latent_length).to(dev)
net.load_state_dict(torch.load(f"results/model0{epoch}.pt"))
net.eval()

# Get intermediate representation
# Get the first encoder block
def shot_forward(model, data):
    x = model.data_compress(data)
    x = model.encoder1(x)
    x = model.encoder2(x)
    x = model.downsample2(x)
    x = model.encoder3(x)
    x = model.downsample3(x)
    return x

shot_latent = shot_forward(net, obs)
print(shot_latent.shape)
fig,axes=plt.subplots(4,4,figsize=(6,6))
for i in range(16):
    ax=axes[i//4,i%4]
    ax.imshow(shot_latent[0,i].detach().cpu().numpy(),cmap='jet', aspect='auto')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"figures/shot_latent_e{epoch}.png")
plt.show()

