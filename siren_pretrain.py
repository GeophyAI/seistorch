import numpy as np
import matplotlib.pyplot as plt
import torch
from wavetorch.siren import Siren
from wavetorch.model import build_model

config_file = r"./config/fixed_acoustic.yml"
PMLN = 50
siren = Siren(in_features=2, out_features=1, hidden_features=128, 
                           hidden_layers=4, outermost_linear=True)
siren.cuda()
cfg, model = build_model(config_file, device="cuda:0", mode="inversion")
init_model = np.load("/home/les_01/wangsw/models/marmousi/elastic_modified/linear_vp_z12.5m_x12.5m_expand.npy")
init_model = np.pad(init_model, ((PMLN,PMLN),(PMLN,PMLN)), 'edge')/1000.
"""Initial model"""
plt.imshow(init_model)
plt.title("Initial Model")
plt.show()
"""Pretrain the model"""
init_model = torch.from_numpy(init_model).float().cuda()
init_model = init_model.reshape(-1, 1)

optim = torch.optim.Adam(lr=1e-4, params=siren.parameters())
total_steps = 1500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 25
shape = model.cell.geom.domain_shape
for step in range(total_steps):
    model_output, coords = siren(model.cell.geom.coords)    
    loss = ((model_output - init_model)**2).mean()
    
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        # img_grad = gradient(model_output, coords)
        # img_laplacian = laplace(model_output, coords)

        fig, axes = plt.subplots(1,2, figsize=(6,4))
        axes[0].imshow(model_output.cpu().view(shape).detach().numpy(), aspect="auto")
        axes[1].plot(model_output.cpu().view(shape).detach().numpy()[:,500], label="predicted")
        axes[1].plot(init_model.cpu().view(shape).detach().numpy()[:,500], label="target")
        plt.legend()
        # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(281,1561).detach().numpy())
        # axes[2].imshow(img_laplacian.cpu().view(281,1561).detach().numpy())
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()

"""Save the pretrain model"""
torch.save(siren.state_dict(), "/home/les_01/wangsw/fwi/linear_vp_z12.5m_x12.5m_expand.pt")