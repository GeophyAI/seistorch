
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from wavetorch.utils import diff_using_roll, ricker_wave, to_tensor
from wavetorch.utils import save_boundaries, restore_boundaries
from wavetorch.cell_elastic import _time_step

import torch

NPML = 50
N = 2

def _time_step_backward(*args):

    vp, vs, rho = args[0:3]
    p, vx, vz, txx, tzz, txz = args[3:9]
    dt, h, d = args[9:12]
    p_bd, vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd = args[-1]

    # vp = vp.unsqueeze(0)
    # vs = vs.unsqueeze(0)
    # rho = rho.unsqueeze(0)
    # d = d.unsqueeze(0)

    """Update velocity components"""
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    # Define the region where the computation is performed
    compute_region = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region]
    vx, vz = vx_copy[compute_region], vz_copy[compute_region]
    d = d[compute_region]

    # The rest of your computation code...
    txx_x = diff_using_roll(txx, 2)[compute_region]
    txz_z = diff_using_roll(txz, 1, False)[compute_region]
    tzz_z = diff_using_roll(tzz, 1, False)[compute_region]
    txz_x = diff_using_roll(txz, 2, False)[compute_region]
    p_x = diff_using_roll(p, 2)[compute_region]
    p_z = diff_using_roll(p, 1, False)[compute_region]

    c = 0.5*dt*d

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd, NPML, N)
    vz_copy = restore_boundaries(vz_copy, vz_bd, NPML, N)

    # Create a copy of the original tensors
    p_copy, txx_copy, tzz_copy, txz_copy = p.clone(), txx.clone(), tzz.clone(), txz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region], lame_mu[compute_region]
    p, vx, vz = p_copy[compute_region], vx_copy[compute_region], vz_copy[compute_region]
    txx, tzz, txz = txx_copy[compute_region], tzz_copy[compute_region], txz_copy[compute_region]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    y_txx = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vx_x-vz_z)+(1-c)*txx)
    y_tzz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_z-vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)
    y_p = (1+c)**-1*(dt*(lame_lambda+lame_mu)*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)


    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region] = y_txx[(slice(None), slice(N,-N), slice(N,-N))]
    tzz_copy[update_region] = y_tzz[(slice(None), slice(N,-N), slice(N,-N))]
    txz_copy[update_region] = y_txz[(slice(None), slice(N,-N), slice(N,-N))]
    p_copy[update_region] = y_p[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    txx_copy = restore_boundaries(txx_copy, txx_bd, NPML, N)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd, NPML, N)
    txz_copy = restore_boundaries(txz_copy, txz_bd, NPML, N)
    p_copy = restore_boundaries(p_copy, p_bd, NPML, N)

    return p_copy, vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy
    
# def _time_step(vp, vs, rho, p, vx, vz, txx, tzz, txz, dt, h, d):
def _time_step(*args):

    vp, vs, rho = args[:3]
    p, vx, vz, txx, tzz, txz = args[3:-3]
    dt, h, d = args[-3:]

    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    """Calculate partial derivative of velocity components"""
    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    c = 0.5*dt*d

    y_txx = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vx_x-vz_z)+(1-c)*txx)

    y_tzz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_z-vx_x)+(1-c)*tzz)

    y_txz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = diff_using_roll(y_txx, 2)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1, False)
    txz_x = diff_using_roll(y_txz, 2, False)

    y_p = (1+c)**-1*(-dt*(lame_lambda+lame_mu)*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)

    p_x = diff_using_roll(y_p, 2)
    p_z = diff_using_roll(y_p, 1, False)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    return y_p, y_vx, y_vz, y_txx, y_tzz, y_txz

dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

pmld = np.load("/home/wangsw/Desktop/wangsw/fwi/pmld.npy")
pmld = np.expand_dims(pmld, 0)
pmld = torch.from_numpy(pmld).to(dev)
nt = 2001

fm = 5
dt = to_tensor(0.001)
dt2 = 0.001
dh = to_tensor(10).to(dev)
src_x = pmld.shape[2]//2
src_z = 53#pmld.shape[1]//2

vp = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vp.npy")#torch.ones_like(pmld).to(dev)*2500
vs = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vs.npy")#torch.ones_like(pmld).to(dev)*2500/1.73
src = ricker_wave(fm, dt2, nt).to(dev)

vp =np.pad(vp, ((NPML,NPML), (NPML,NPML)), mode="edge")
vp = np.expand_dims(vp, 0)
vp = to_tensor(vp).to(dev)

vs =np.pad(vs, ((NPML,NPML), (NPML,NPML)), mode="edge")
vs = np.expand_dims(vs, 0)
vs = to_tensor(vs).to(dev)

rho = torch.ones_like(pmld).to(dev)*2000

"""Forward"""
boundarys = []
wavefield = [torch.zeros_like(pmld).to(dev) for _ in range(6)]
save_for_backward = []
forwards = []
for i in range(nt):
    wavefield = _time_step(vp, vs, rho, *wavefield, dt, dh, pmld)
    wavefield[0][...,src_z,src_x] += src[i]

    boundarys.append([save_boundaries(_wavefield.cpu(), NPML, N) for _wavefield in wavefield])
    # print([b.size() for b in boundarys[i][0]])

    if i %400==0:
        forwards.append(wavefield[1][0][50:-50,50:-50].cpu().detach().numpy())

    if i==nt-1:
        save_for_backward.append(wavefield)
# wavefield = [torch.zeros_like(pmld) for _ in range(5)]
print("Backward")
wavefield = list(save_for_backward[0])
backwards = []
"Backward"
for i in range(nt-1, -1, -1):
    wavefield = _time_step_backward(vp, vs, rho, *wavefield, dt, dh, pmld, boundarys[i])

    if i%400==0:
        backwards.append(wavefield[1][0][50:-50,50:-50].cpu().detach().numpy())


fig, axes = plt.subplots(1,3, figsize=(8,3))
no=3
_forward = forwards[no]
_backward = backwards[5-no]
shows = [_forward, _backward, _forward-_backward]
vmin,vmax=np.percentile(shows[0], [2,98])
for d, ax in zip(shows, axes.ravel()):
    ax.imshow(d, vmin=vmin, vmax=vmax, cmap=plt.cm.gray, aspect="auto")
plt.show()


        