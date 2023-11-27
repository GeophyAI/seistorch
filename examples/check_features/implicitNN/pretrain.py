import numpy as np
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append("../../..")

from seistorch.networks import Siren, CNN, Encoder
from seistorch.model import build_model

nntype = 'encoder'

config_file = f"./config/{nntype}.yml"
PMLN = 50
unit = 0.001
# siren = Siren(in_features=2, out_features=1, hidden_features=128,
#                            hidden_layers=4, outermost_linear=True)
# siren.cuda()
cfg, model = build_model(config_file, device="cuda:0", mode="inversion")
init_model = np.load("../../models/marmousi_model/linear_vp.npy")
init_model = np.pad(init_model, ((PMLN,PMLN),(PMLN,PMLN)), 'edge')*unit
# init_model = np.ones_like(init_model)*1500.
kwargs_nn = {"in_features":cfg['training']['implicit']['in_features'],
                "out_features":cfg['training']['implicit']['out_features'],
                "hidden_features":cfg['training']['implicit']['hidden_features'],
                "hidden_layers":cfg['training']['implicit']['hidden_layers'],
                "outermost_linear":True,
                "domain_shape":init_model.shape}


nn = {"siren":Siren, "cnn":CNN, "encoder":Encoder}[nntype](**kwargs_nn).to("cuda:0")
print(f"Number of parameters: {sum(p.numel() for p in nn.parameters())}")
# Normalization using std and mean
mean = 3000*unit
std = 1000*unit
target_model = (init_model - mean) / std
# target_model = init_model
print(f"Mean: {init_model.mean()}, Std: {init_model.std()}")
"""Initial model"""
plt.imshow(init_model)
plt.title("Initial Model")
plt.show()
"""Pretrain the model"""
target_model = torch.from_numpy(target_model).float().cuda()
# target_model = target_model.reshape(-1, 1)

optim = torch.optim.Adam(lr=1e-4, params=nn.parameters())
total_steps = 501 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 25
shape = model.cell.geom.domain_shape
rand_vector = torch.rand(cfg['training']['implicit']['in_features']).float().cuda()
for step in range(total_steps):
    if nntype == 'siren':
        model_output, _ = nn(nn.coords.to("cuda:0"))
    else:
        model_output = nn(rand_vector)
    # model_output = model_output*3000+1000.
    loss = ((model_output - target_model)**2).mean()
    model_output = model_output*std+mean
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        print(f"Max: {model_output.max()}, Min: {model_output.min()}")
        # img_grad = gradient(model_output, coords)
        # img_laplacian = laplace(model_output, coords)
        fig, axes = plt.subplots(1,2, figsize=(6,4))
        axes[0].imshow(model_output.cpu().view(shape).detach().numpy(), aspect="auto")
        axes[1].plot(model_output.cpu().view(shape).detach().numpy()[:,100], label="predicted")
        axes[1].plot(init_model.reshape(shape)[:,100], label="target")
        plt.legend()
        # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(281,1561).detach().numpy())
        # axes[2].imshow(img_laplacian.cpu().view(281,1561).detach().numpy())
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()

"""Save the pretrain model"""
import os
os.makedirs("./pretrained", exist_ok=True)
torch.save(nn.state_dict(), f"./pretrained/{nntype}.pt")
print("Model saved!")