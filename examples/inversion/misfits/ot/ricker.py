import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('../../../..')
from seistorch.loss import L2, Wasserstein1d, NormalizedIntegrationMethod, Integration

def ricker(fm, dt, T, delay):
    t = np.arange(0, T, dt)
    t -= delay
    y = (1.0 - 2.0 * (np.pi ** 2) * (fm ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (fm ** 2) * (t ** 2))
    return y

def to_tensor(d):
    d = np.reshape(d, (1, len(d), 1, 1))
    return torch.tensor(d).float()

dt = 0.001
fm = 5
T = 3
delays = np.linspace(-1.25, 1.25, 501)
# L2
loss = []
methods = [L2(), 
           Wasserstein1d('linear'), 
           Wasserstein1d('square'), 
           Wasserstein1d('exp'), 
           Wasserstein1d('abs'),
           NormalizedIntegrationMethod(), 
           Integration()]
for delay in delays:
    temp = []
    s1 = ricker(fm, dt, T, T/2+delay)
    s2 = ricker(fm, dt, T, T/2)
    for method in methods:
        temp.append(method(to_tensor(s1), to_tensor(s2)).item())
    loss.append(temp)
lossnames = ['L2', 'WD-linear', 'WD-square', 'WD-exp', 'WD-abs', 'NIM-Square', 'Integration']
loss = np.array(loss)
fig, axes = plt.subplots(2, len(lossnames)//2, figsize=(8, 6))
for i, (name, ax) in enumerate(zip(lossnames, axes.ravel())):
    ax.plot(delays, loss[:, i], label=name)
    ax.set_xlabel('Delay (s)')
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
plt.savefig('figures/loss_vs_delay.png', dpi=300, bbox_inches='tight')
plt.show()

