import numpy as np
import torch

model = torch.load('results/model_F00E02.pt')['rz'].cpu().detach().numpy()

print(model.max(), model.min())