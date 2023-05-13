
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from wavetorch.utils import diff_using_roll, ricker_wave, to_tensor
from wavetorch.cell_elastic import _time_step
from wavetorch.operators import _laplacian
from wavetorch.utils import save_boundaries, restore_boundaries

import torch

def _time_step(c, y1, y2, dt, h, b):
    # Equation S8(S9)
    # When b=0, without boundary conditon.
    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * y1 - torch.mul((dt**-2 - b * dt**-1), y2)
                + torch.mul(c.pow(2), _laplacian(y1, h)))
                )
    return y

NPML = 50
N = 5

def _time_step_backward(c, y1, y2, dt, h, b):

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    y1_copy, y2_copy = y1.clone(), y2.clone()

    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * y1 - torch.mul((dt**-2 - b * dt**-1), y2)
                + torch.mul(c.pow(2), _laplacian(y1, h)))
                )
    return y

dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

pmld = np.load("/home/wangsw/Desktop/wangsw/fwi/pmld.npy")
pmld = np.expand_dims(pmld, 0)
pmld = torch.from_numpy(pmld).to(dev)
nt = 2001
pmln = 50
order = 2
fm = 5
dt = to_tensor(0.001)
dt2 = 0.001
dh = to_tensor(10).to(dev)
src_x = pmld.shape[2]//2
src_z = 53#pmld.shape[1]//2

vp = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vp.npy")#torch.ones_like(pmld).to(dev)*2500
src = ricker_wave(fm, dt2, nt).to(dev)

vp =np.pad(vp, ((pmln,pmln), (pmln,pmln)), mode="edge")
vp = np.expand_dims(vp, 0)
vp = to_tensor(vp).to(dev)


"""Forward"""
boundarys = []
wavefield = [torch.zeros_like(pmld).to(dev) for _ in range(2)]
save_for_backward = []
forwards = []

"""forward"""
for i in range(nt):

    y = _time_step(vp,  *wavefield, dt, dh, pmld)
    temp = wavefield[0].clone()
    wavefield.clear()
    wavefield.extend([y, temp])

    """Add source"""
    wavefield[1][...,src_z,src_x] += src[i]

    boundarys.append([save_boundaries(_wavefield.cpu(), pmln, order) for _wavefield in wavefield])

    if i %400==0:
        forwards.append(wavefield[1][0][50:-50,50:-50].cpu().detach().numpy())

    if i==nt-1 and i== nt-2:
        save_for_backward.append(wavefield)


"""backward"""
for i in range(nt-1, -1, -1):
#     wavefield = _time_step_backward(vp, *wavefield, dt, dh, pmld, boundarys[i])

#     if i %400==0:
#         backwards.append(wavefield[1][0][50:-50,50:-50].cpu().detach().numpy())

"""Show the results"""
fig, axes = plt.subplots(1,3, figsize=(8,3))
no=3
_forward = forwards[no]
_backward = forwards[5-no]
shows = [_forward, _backward, _forward-_backward]
vmin,vmax=np.percentile(shows[0], [2,98])
for d, ax in zip(shows, axes.ravel()):
    ax.imshow(d, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect="auto")
plt.show()

