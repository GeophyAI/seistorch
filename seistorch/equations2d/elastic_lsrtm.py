import torch
from .utils import diff_using_roll, restore_boundaries

NPML = 49
N = 1

def _time_step(*args, **kwargs):

    vp, vs, rho, rvp, rvs, rrho = args[0:6]
    vx, vz, txx, tzz, txz, vxs, vzs, txxs, tzzs, txzs = args[6:16]
    dt, h, d = args[16:19]
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    ### Step1: Calculate background wavefield ############
    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    c = 0.5*dt*d

    # Update the stress components
    y_txx  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = diff_using_roll(y_txx, 2, False)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1)
    txz_x = diff_using_roll(y_txz, 2)

    # Update the velocity components
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    #####################################################
    ### Step2: Calculate scattered wavefield ############
    #####################################################
    vxs_x = diff_using_roll(vxs, 2)
    vzs_z = diff_using_roll(vzs, 1, False)
    vxs_z = diff_using_roll(vxs, 1)
    vzs_x = diff_using_roll(vzs, 2, False)

    # y_txxs  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*rvp*vx_x \
    #                                    +((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vz_z \
    #                                    +lame_lambda*(vxs_x+vzs_z) \
    #                                    +2*lame_mu*vxs_x)+(1-c)*txxs)
    # y_tzzs  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*rvp*vz_z \
    #                                    +((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vx_x \
    #                                    +lame_lambda*(vxs_x+vzs_z) \
    #                                    +2*lame_mu*vzs_z)+(1-c)*tzzs)
    # y_txzs = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(rvs*(vx_z+vz_x) \
    #                                           +(vxs_z+vzs_x))+(1-c)*txzs)

    y_txxs  = (1+c)**-1*(dt*h.pow(-1)*(2*(lame_lambda+2*lame_mu)*rvp*vx_x \
                                       +2*((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vz_z \
                                       +lame_lambda*(vxs_x+vzs_z) \
                                       +2*lame_mu*vxs_x)+(1-c)*txxs)
    y_tzzs  = (1+c)**-1*(dt*h.pow(-1)*(2*(lame_lambda+2*lame_mu)*rvp*vz_z \
                                       +2*((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vx_x \
                                       +lame_lambda*(vxs_x+vzs_z) \
                                       +2*lame_mu*vzs_z)+(1-c)*tzzs)
    y_txzs = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(2*rvs*(vx_z+vz_x) \
                                              +(vxs_z+vzs_x))+(1-c)*txzs)
    
    # Update the velocity components
    txxs_x = diff_using_roll(y_txxs, 2, False)
    txzs_z = diff_using_roll(y_txzs, 1, False)
    tzzs_z = diff_using_roll(y_tzzs, 1)
    txzs_x = diff_using_roll(y_txzs, 2)

    y_vxs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txxs_x+txzs_z)+(1-c)*vxs)
    y_vzs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txzs_x+tzzs_z)+(1-c)*vzs)

    return y_vx, y_vz, y_txx, y_tzz, y_txz, y_vxs, y_vzs, y_txxs, y_tzzs, y_txzs

def _time_step_backward(*args, **kwargs):

    vp, vs, rho, rvp, rvs, rrho = args[0:6]
    vx, vz, txx, tzz, txz, vxs, vzs, txxs, tzzs, txzs = args[6:16]
    dt, h, d = args[16:19]
    vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd, vxs_bd, vzs_bd, txxs_bd, tzzs_bd, txzs_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    vs = vs.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)
    rvp = rvp.unsqueeze(0)
    rvs = rvs.unsqueeze(0)

    """Update velocity components"""
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()
    vxs_copy, vzs_copy = vxs.clone(), vzs.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    vxs, vzs = vxs_copy[compute_region_slice], vzs_copy[compute_region_slice]
    d = d[compute_region_slice]

    # calculate the background stress components
    txx_x = diff_using_roll(txx, 2, False)[compute_region_slice]
    txz_z = diff_using_roll(txz, 1, False)[compute_region_slice]
    tzz_z = diff_using_roll(tzz, 1)[compute_region_slice]
    txz_x = diff_using_roll(txz, 2)[compute_region_slice]

    # calculate the scattered stress components
    txxs_x = diff_using_roll(txxs, 2, False)[compute_region_slice]
    txzs_z = diff_using_roll(txzs, 1, False)[compute_region_slice]
    tzzs_z = diff_using_roll(tzzs, 1)[compute_region_slice]
    txzs_x = diff_using_roll(txzs, 2)[compute_region_slice]

    c = 0.5*dt*d

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    y_vxs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txxs_x+txzs_z)+(1-c)*vxs)
    y_vzs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txzs_x+tzzs_z)+(1-c)*vzs)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]
    vxs_copy[update_region_slice] = y_vxs[(slice(None), slice(N,-N), slice(N,-N))]
    vzs_copy[update_region_slice] = y_vzs[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd)
    vz_copy = restore_boundaries(vz_copy, vz_bd)
    vxs_copy = restore_boundaries(vxs_copy, vxs_bd)
    vzs_copy = restore_boundaries(vzs_copy, vzs_bd)

    # Create a copy of the original tensors
    txx_copy, tzz_copy, txz_copy = txx.clone(), tzz.clone(), txz.clone()
    txxs_copy, tzzs_copy, txzs_copy = txxs.clone(), tzzs.clone(), txzs.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region_slice], lame_mu[compute_region_slice]
    rvp, rvs = rvp[compute_region_slice], rvs[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    vxs, vzs = vxs_copy[compute_region_slice], vzs_copy[compute_region_slice]
    txx, tzz, txz = txx_copy[compute_region_slice], tzz_copy[compute_region_slice], txz_copy[compute_region_slice]
    txxs, tzzs, txzs = txxs_copy[compute_region_slice], tzzs_copy[compute_region_slice], txzs_copy[compute_region_slice]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    vxs_x = diff_using_roll(vxs, 2)
    vzs_z = diff_using_roll(vzs, 1, False)
    vxs_z = diff_using_roll(vxs, 1)
    vzs_x = diff_using_roll(vzs, 2, False)

    y_txx  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    y_txxs  = (1+c)**-1*(dt*h.pow(-1)*(2*(lame_lambda+2*lame_mu)*rvp*vx_x \
                                       +2*((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vz_z \
                                       +lame_lambda*(vxs_x+vzs_z) \
                                       +2*lame_mu*vxs_x)+(1-c)*txxs)
    y_tzzs  = (1+c)**-1*(dt*h.pow(-1)*(2*(lame_lambda+2*lame_mu)*rvp*vz_z \
                                       +2*((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vx_x \
                                       +lame_lambda*(vxs_x+vzs_z) \
                                       +2*lame_mu*vzs_z)+(1-c)*tzzs)
    y_txzs = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(2*rvs*(vx_z+vz_x) \
                                              +(vxs_z+vzs_x))+(1-c)*txzs)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region_slice] = y_txx[(slice(None), slice(N,-N), slice(N,-N))]
    tzz_copy[update_region_slice] = y_tzz[(slice(None), slice(N,-N), slice(N,-N))]
    txz_copy[update_region_slice] = y_txz[(slice(None), slice(N,-N), slice(N,-N))]
    txxs_copy[update_region_slice] = y_txxs[(slice(None), slice(N,-N), slice(N,-N))]
    tzzs_copy[update_region_slice] = y_tzzs[(slice(None), slice(N,-N), slice(N,-N))]
    txzs_copy[update_region_slice] = y_txzs[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    # with torch.no_grad():
    txx_copy = restore_boundaries(txx_copy, txx_bd)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd)
    txz_copy = restore_boundaries(txz_copy, txz_bd)
    txxs_copy = restore_boundaries(txxs_copy, txxs_bd)
    tzzs_copy = restore_boundaries(tzzs_copy, tzzs_bd)
    txzs_copy = restore_boundaries(txzs_copy, txzs_bd)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var.data.copy_(src_func(source_var, src_values, -1))

    return vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy, vxs_copy, vzs_copy, txxs_copy, tzzs_copy, txzs_copy

def _time_step_backward_multiple(*args, **kwargs):
    top=0

    vp, vs, rho, rvp, rvs, rrho = args[0:6]
    vx, vz, txx, tzz, txz, vxs, vzs, txxs, tzzs, txzs = args[6:16]
    dt, h, d = args[16:19]
    vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd, vxs_bd, vzs_bd, txxs_bd, tzzs_bd, txzs_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    vs = vs.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)
    rvp = rvp.unsqueeze(0)
    rvs = rvs.unsqueeze(0)

    """Update velocity components"""
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(top, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(top, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()
    vxs_copy, vzs_copy = vxs.clone(), vzs.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    vxs, vzs = vxs_copy[compute_region_slice], vzs_copy[compute_region_slice]
    d = d[compute_region_slice]

    # calculate the background stress components
    txx_x = diff_using_roll(txx, 2, False)[compute_region_slice]
    txz_z = diff_using_roll(txz, 1, False)[compute_region_slice]
    tzz_z = diff_using_roll(tzz, 1)[compute_region_slice]
    txz_x = diff_using_roll(txz, 2)[compute_region_slice]

    # calculate the scattered stress components
    txxs_x = diff_using_roll(txxs, 2, False)[compute_region_slice]
    txzs_z = diff_using_roll(txzs, 1, False)[compute_region_slice]
    tzzs_z = diff_using_roll(tzzs, 1)[compute_region_slice]
    txzs_x = diff_using_roll(txzs, 2)[compute_region_slice]

    c = 0.5*dt*d

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    y_vxs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txxs_x+txzs_z)+(1-c)*vxs)
    y_vzs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txzs_x+tzzs_z)+(1-c)*vzs)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(top,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(top,-N), slice(N,-N))]
    vxs_copy[update_region_slice] = y_vxs[(slice(None), slice(top,-N), slice(N,-N))]
    vzs_copy[update_region_slice] = y_vzs[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd)
    vz_copy = restore_boundaries(vz_copy, vz_bd)
    vxs_copy = restore_boundaries(vxs_copy, vxs_bd)
    vzs_copy = restore_boundaries(vzs_copy, vzs_bd)

    # Create a copy of the original tensors
    txx_copy, tzz_copy, txz_copy = txx.clone(), tzz.clone(), txz.clone()
    txxs_copy, tzzs_copy, txzs_copy = txxs.clone(), tzzs.clone(), txzs.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region_slice], lame_mu[compute_region_slice]
    rvp, rvs = rvp[compute_region_slice], rvs[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    vxs, vzs = vxs_copy[compute_region_slice], vzs_copy[compute_region_slice]
    txx, tzz, txz = txx_copy[compute_region_slice], tzz_copy[compute_region_slice], txz_copy[compute_region_slice]
    txxs, tzzs, txzs = txxs_copy[compute_region_slice], tzzs_copy[compute_region_slice], txzs_copy[compute_region_slice]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    vxs_x = diff_using_roll(vxs, 2)
    vzs_z = diff_using_roll(vzs, 1, False)
    vxs_z = diff_using_roll(vxs, 1)
    vzs_x = diff_using_roll(vzs, 2, False)

    y_txx  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    y_txxs  = (1+c)**-1*(dt*h.pow(-1)*(2*(lame_lambda+2*lame_mu)*rvp*vx_x \
                                       +2*((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vz_z \
                                       +lame_lambda*(vxs_x+vzs_z) \
                                       +2*lame_mu*vxs_x)+(1-c)*txxs)
    y_tzzs  = (1+c)**-1*(dt*h.pow(-1)*(2*(lame_lambda+2*lame_mu)*rvp*vz_z \
                                       +2*((lame_lambda+2*lame_mu)*rvp-2*lame_mu*rvs)*vx_x \
                                       +lame_lambda*(vxs_x+vzs_z) \
                                       +2*lame_mu*vzs_z)+(1-c)*tzzs)
    y_txzs = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(2*rvs*(vx_z+vz_x) \
                                              +(vxs_z+vzs_x))+(1-c)*txzs)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region_slice] = y_txx[(slice(None), slice(top,-N), slice(N,-N))]
    tzz_copy[update_region_slice] = y_tzz[(slice(None), slice(top,-N), slice(N,-N))]
    txz_copy[update_region_slice] = y_txz[(slice(None), slice(top,-N), slice(N,-N))]
    txxs_copy[update_region_slice] = y_txxs[(slice(None), slice(top,-N), slice(N,-N))]
    tzzs_copy[update_region_slice] = y_tzzs[(slice(None), slice(top,-N), slice(N,-N))]
    txzs_copy[update_region_slice] = y_txzs[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore the boundary
    txx_copy = restore_boundaries(txx_copy, txx_bd)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd)
    txz_copy = restore_boundaries(txz_copy, txz_bd)
    txxs_copy = restore_boundaries(txxs_copy, txxs_bd)
    tzzs_copy = restore_boundaries(tzzs_copy, tzzs_bd)
    txzs_copy = restore_boundaries(txzs_copy, txzs_bd)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var.data.copy_(src_func(source_var, src_values, -1))

    return vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy, vxs_copy, vzs_copy, txxs_copy, tzzs_copy, txzs_copy
