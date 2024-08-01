# Author: Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import matplotlib as mpl
import torch
from scipy.signal import hilbert

from ot.lp import wasserstein_1d
from ot.datasets import make_1D_gauss as gauss
from ot.utils import proj_simplex

red = np.array(mpl.colors.to_rgb('red'))
blue = np.array(mpl.colors.to_rgb('blue'))

def ricker(fm=10, t=1, dt=0.001, delay=0.1):
    t = np.linspace(0, t, int(t//dt)+1)
    t -= delay
    y = (1.0 - 2.0 * (np.pi ** 2) * (fm ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (fm ** 2) * (t ** 2))
    return y

n = 1000  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
# a = gauss(n, m=20, s=5)  # m= mean, s= std
# b = gauss(n, m=60, s=10)

a = ricker(delay=0.3)
b = ricker(delay=0.6)

a = np.abs(hilbert(a))
b = np.abs(hilbert(b))

# enforce sum to one on the support
a = a / a.sum()
b = b / b.sum()

device = "cuda" if torch.cuda.is_available() else "cpu"

# use pyTorch for our data
x_torch = torch.tensor(x).to(device=device)
a_torch = torch.tensor(a).to(device=device).requires_grad_(True)
b_torch = torch.tensor(b).to(device=device)

lr = 5e-7
nb_iter_max = 800

loss_iter = []

pl.figure(1, figsize=(8, 4))
pl.plot(x, a, 'b', label='Source distribution')
pl.plot(x, b, 'r', label='Target distribution')

for i in range(nb_iter_max):
    # Compute the Wasserstein 1D with torch backend
    loss = wasserstein_1d(x_torch, x_torch, a_torch, b_torch, p=1)
    # record the corresponding loss value
    loss_iter.append(loss.clone().detach().cpu().numpy())
    loss.backward()

    # performs a step of projected gradient descent
    with torch.no_grad():
        grad = a_torch.grad
        a_torch -= a_torch.grad * lr  # step
        a_torch.grad.zero_()
        a_torch.data = proj_simplex(a_torch)  # projection onto the simplex

    # plot one curve every 10 iterations
    if i % 10 == 0:
        mix = float(i) / nb_iter_max
        pl.plot(x, a_torch.clone().detach().cpu().numpy(), c=(1 - mix) * blue + mix * red)

pl.legend()
pl.title('Distribution along the iterations of the projected gradient descent')
pl.show()

pl.figure(2)
pl.plot(range(nb_iter_max), loss_iter, lw=3)
pl.title('Evolution of the loss along iterations', fontsize=16)
pl.show()