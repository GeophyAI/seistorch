import torch
from .utils import diff_using_roll, restore_boundaries

NPML = 49
N = 1

def _time_step(*args, **kwargs):
    vp, vs, rho = args[0:3]
    vx, vz, txx, tzz, txz = args[3:8]
    dt, h, d = args[8:11]
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))
    c = 0.5*dt*d

    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    # Equation A-8
    y_txx  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    # Equation A-9
    y_tzz  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    # Equation A-10
    y_txz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = diff_using_roll(y_txx, 2, False)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1)
    txz_x = diff_using_roll(y_txz, 2)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz

def _time_step_backward(*args, **kwargs):

    vp, vs, rho = args[0:3]
    vx, vz, txx, tzz, txz = args[3:8]
    dt, h, d = args[8:11]
    vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    vs = vs.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)

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
    txx_x = diff_using_roll(txx, 2, False)[compute_region_slice]
    txz_z = diff_using_roll(txz, 1, False)[compute_region_slice]
    tzz_z = diff_using_roll(tzz, 1)[compute_region_slice]
    txz_x = diff_using_roll(txz, 2)[compute_region_slice]

    c = 0.5*dt*d

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd)
    vz_copy = restore_boundaries(vz_copy, vz_bd)

    # Create a copy of the original tensors
    txx_copy, tzz_copy, txz_copy = txx.clone(), tzz.clone(), txz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region_slice], lame_mu[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    txx, tzz, txz = txx_copy[compute_region_slice], tzz_copy[compute_region_slice], txz_copy[compute_region_slice]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    c = 0.5*dt*d
    y_txx  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region_slice] = y_txx[(slice(None), slice(N,-N), slice(N,-N))]
    tzz_copy[update_region_slice] = y_tzz[(slice(None), slice(N,-N), slice(N,-N))]
    txz_copy[update_region_slice] = y_txz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    txx_copy = restore_boundaries(txx_copy, txx_bd)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd)
    txz_copy = restore_boundaries(txz_copy, txz_bd)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var.data.copy_(src_func(source_var, src_values, -1))

        # source_var = src_func(source_var, src_values, -1)

    return vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy

def _time_step_backward_multiple(*args, **kwargs):
    top=0
    vp, vs, rho = args[0:3]
    vx, vz, txx, tzz, txz = args[3:8]
    dt, h, d = args[8:11]
    vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    vs = vs.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)

    """Update velocity components"""
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(top, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(top, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    d = d[compute_region_slice]

    # The rest of your computation code...
    txx_x = diff_using_roll(txx, 2, False)[compute_region_slice]
    txz_z = diff_using_roll(txz, 1, False)[compute_region_slice]
    tzz_z = diff_using_roll(tzz, 1)[compute_region_slice]
    txz_x = diff_using_roll(txz, 2)[compute_region_slice]

    c = 0.5*dt*d

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(top,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd, multiple=True)
    vz_copy = restore_boundaries(vz_copy, vz_bd, multiple=True)

    # Create a copy of the original tensors
    txx_copy, tzz_copy, txz_copy = txx.clone(), tzz.clone(), txz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region_slice], lame_mu[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    txx, tzz, txz = txx_copy[compute_region_slice], tzz_copy[compute_region_slice], txz_copy[compute_region_slice]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    c = 0.5*dt*d
    y_txx  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region_slice] = y_txx[(slice(None), slice(top,-N), slice(N,-N))]
    tzz_copy[update_region_slice] = y_tzz[(slice(None), slice(top,-N), slice(N,-N))]
    txz_copy[update_region_slice] = y_txz[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore the boundary
    txx_copy = restore_boundaries(txx_copy, txx_bd, multiple=True)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd, multiple=True)
    txz_copy = restore_boundaries(txz_copy, txz_bd, multiple=True)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var.data.copy_(src_func(source_var, src_values, -1))

        # source_var = src_func(source_var, src_values, -1)

    return vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy