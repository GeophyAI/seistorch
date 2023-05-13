
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from wavetorch.utils import diff_using_roll, ricker_wave, to_tensor
from wavetorch.cell_elastic import _time_step

import torch

NPML = 50
N = 2

def _time_step_backward(vp, vs, rho, vx, vz, txx, tzz, txz, dt, h, d, memory):

    vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd = memory

    """Update velocity components"""
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    d = d[compute_region_slice]

    # The rest of your computation code...
    txx_x = diff_using_roll(txx, 2)[compute_region_slice]
    txz_z = diff_using_roll(txz, 1, False)[compute_region_slice]
    tzz_z = diff_using_roll(tzz, 1, False)[compute_region_slice]
    txz_x = diff_using_roll(txz, 2, False)[compute_region_slice]

    c = 0.5*dt*d
    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_ring_elements(vx_copy, vx_bd, pmln, order)
    vz_copy = restore_ring_elements(vz_copy, vz_bd, pmln, order)

    # Create a copy of the original tensors
    txx_copy, tzz_copy, txz_copy = txx.clone(), tzz.clone(), txz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region_slice], lame_mu[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    txx, tzz, txz = txx_copy[compute_region_slice], tzz_copy[compute_region_slice], txz_copy[compute_region_slice]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    c = 0.5*dt*d
    y_txx  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region_slice] = y_txx[(slice(None), slice(N,-N), slice(N,-N))]
    tzz_copy[update_region_slice] = y_tzz[(slice(None), slice(N,-N), slice(N,-N))]
    txz_copy[update_region_slice] = y_txz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    txx_copy = restore_ring_elements(txx_copy, txx_bd, pmln, order)
    tzz_copy = restore_ring_elements(tzz_copy, tzz_bd, pmln, order)
    txz_copy = restore_ring_elements(txz_copy, txz_bd, pmln, order)

    return vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy

def _time_step_vel_backward(rho, vx, vz, txx, tzz, txz, dt, h, d):
    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    txx, tzz, txz = txx[compute_region_slice], tzz[compute_region_slice], txz[compute_region_slice]
    d = d[compute_region_slice]

    # The rest of your computation code...
    txx_x = diff_using_roll(txx, 2)
    txz_z = diff_using_roll(txz, 1, False)
    tzz_z = diff_using_roll(tzz, 1, False)
    txz_x = diff_using_roll(txz, 2, False)

    c = 0.5*dt*d
    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]

    return vx_copy, vz_copy


def _time_step_stress_backward(vp, vs, rho, vx, vz, txx, tzz, txz, dt, h, d):
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))
    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    txx_copy, tzz_copy, txz_copy = txx.clone(), tzz.clone(), txz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region_slice], lame_mu[compute_region_slice]
    vx, vz = vx[compute_region_slice], vz[compute_region_slice]
    txx, tzz, txz = txx_copy[compute_region_slice], tzz_copy[compute_region_slice], txz_copy[compute_region_slice]
    d = d[compute_region_slice]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    c = 0.5*dt*d
    y_txx  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region_slice] = y_txx[(slice(None), slice(N,-N), slice(N,-N))]
    tzz_copy[update_region_slice] = y_tzz[(slice(None), slice(N,-N), slice(N,-N))]
    txz_copy[update_region_slice] = y_txz[(slice(None), slice(N,-N), slice(N,-N))]

    return txx_copy, tzz_copy, txz_copy


def save_ring_elements(tensor, NPML, N):
    tensor = tensor.squeeze(0)
    top = tensor[NPML:NPML+N, :]
    bottom = tensor[-(NPML+N):-NPML, :]
    left = tensor[:,NPML:NPML+N]
    right = tensor[:, -(NPML+N):-NPML]

    return top, bottom, left, right

def restore_ring_elements(tensor, memory, NPML, N):

    top, bottom, left, right = memory
    tensor[..., NPML:NPML+N, :] = top
    tensor[..., -(NPML+N):-NPML, :] = bottom
    tensor[..., NPML:NPML+N] = left
    tensor[..., -(NPML+N):-NPML] = right
    
    return tensor

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
vs = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vs.npy")#torch.ones_like(pmld).to(dev)*2500/1.73
src = ricker_wave(fm, dt2, nt).to(dev)

vp =np.pad(vp, ((pmln,pmln), (pmln,pmln)), mode="edge")
vp = np.expand_dims(vp, 0)
vp = to_tensor(vp).to(dev)

vs =np.pad(vs, ((pmln,pmln), (pmln,pmln)), mode="edge")
vs = np.expand_dims(vs, 0)
vs = to_tensor(vs).to(dev)

rho = torch.ones_like(pmld).to(dev)*2000

"""Forward"""
boundarys = []
wavefield = [torch.zeros_like(pmld).to(dev) for _ in range(5)]
save_for_backward = []
forwards = []
for i in range(nt):
    wavefield = _time_step(vp, vs, rho, *wavefield, dt, dh, pmld)
    wavefield[1][...,src_z,src_x] += src[i]

    boundarys.append([save_ring_elements(_wavefield.cpu(), pmln, order) for _wavefield in wavefield])
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

    # for _wavefield in wavefield:
    #     restore_ring_elements(_wavefield, boundarys[i], pmln, order)



    # y_vx, y_vz = _time_step_vel_backward(rho, *wavefield, dt, dh, pmld)

    # # for data in [y_vx, y_)]
    # y_vx = restore_ring_elements(y_vx, boundarys[i][0], pmln, order)
    # y_vz = restore_ring_elements(y_vz, boundarys[i][1], pmln, order)
    # wavefield[slice(0, 2)] = [y_vx, y_vz]


    # y_txx, y_tzz, y_txz = _time_step_stress_backward(vp, vs, rho, *wavefield, dt, dh, pmld)
    
    # y_txx = restore_ring_elements(y_txx, boundarys[i][2], pmln, order)
    # y_tzz = restore_ring_elements(y_tzz, boundarys[i][3], pmln, order)
    # y_txz = restore_ring_elements(y_txz, boundarys[i][4], pmln, order)

    # wavefield[slice(2, 5)] = [y_txx, y_tzz, y_txz]

    # wavefield[1][...,src_z,src_x] -= src[i]

    # wavefield = [y_vx, y_vz, y_txx, y_tzz, y_txz]
    # wavefield = _time_step_backward(vp, vs, rho, *wavefield, dt, dh, pmld, i, *boundarys[i])
    # print([b.size() for b in boundarys[i][0]])
    wavefield = _time_step_backward(vp, vs, rho, *wavefield, dt, dh, pmld, boundarys[i])

    if i %400==0:
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


        