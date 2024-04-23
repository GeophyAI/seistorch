from .utils import restore_boundaries, diff_using_roll

NPML=49
N=1
def _time_step_backward(*args):
    vp, vs, rho = args[0:3]
    p, vx, vz, txx, tzz, txz = args[3:9]
    dt, h, d = args[9:12]
    p_bd, vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    vs = vs.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)

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

    # Create a copy of the original tensors
    p_copy, txx_copy, tzz_copy, txz_copy = p.clone(), txx.clone(), tzz.clone(), txz.clone()

    # # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region], lame_mu[compute_region]
    p, vx, vz = p_copy[compute_region], vx_copy[compute_region], vz_copy[compute_region]
    txx, tzz, txz = txx_copy[compute_region], tzz_copy[compute_region], txz_copy[compute_region]

    c = 0.5*dt*d

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

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
    txx_copy = restore_boundaries(txx_copy, txx_bd, NPML, N, multiple=False)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd, NPML, N, multiple=False)
    txz_copy = restore_boundaries(txz_copy, txz_bd, NPML, N, multiple=False)
    p_copy = restore_boundaries(p_copy, p_bd, NPML, N, multiple=False)

    # The rest of your computation code...
    txx_x = diff_using_roll(y_txx, 2, False)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1)
    txz_x = diff_using_roll(y_txz, 2)

    p_x = diff_using_roll(y_p, 2, False)
    p_z = diff_using_roll(y_p, 1)

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd, NPML, N, multiple=False)
    vz_copy = restore_boundaries(vz_copy, vz_bd, NPML, N, multiple=False)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var = src_func(source_var, src_values, -1)

    return p_copy, vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy

def _time_step_backward_multiple(*args):
    top = 0
    vp, vs, rho = args[0:3]
    p, vx, vz, txx, tzz, txz = args[3:9]
    dt, h, d = args[9:12]
    p_bd, vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    vs = vs.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)

    """Update velocity components"""
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    # Define the region where the computation is performed
    compute_region = (slice(None), slice(top, -NPML), slice(NPML, -NPML))
    update_region = (slice(None), slice(top, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region]
    vx, vz = vx_copy[compute_region], vz_copy[compute_region]
    d = d[compute_region]

    # Create a copy of the original tensors
    p_copy, txx_copy, tzz_copy, txz_copy = p.clone(), txx.clone(), tzz.clone(), txz.clone()

    # # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region], lame_mu[compute_region]
    p, vx, vz = p_copy[compute_region], vx_copy[compute_region], vz_copy[compute_region]
    txx, tzz, txz = txx_copy[compute_region], tzz_copy[compute_region], txz_copy[compute_region]

    c = 0.5*dt*d

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    y_txx = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vx_x-vz_z)+(1-c)*txx)
    y_tzz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_z-vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)
    y_p = (1+c)**-1*(dt*(lame_lambda+lame_mu)*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region] = y_txx[(slice(None), slice(top,-N), slice(N,-N))]
    tzz_copy[update_region] = y_tzz[(slice(None), slice(top,-N), slice(N,-N))]
    txz_copy[update_region] = y_txz[(slice(None), slice(top,-N), slice(N,-N))]
    p_copy[update_region] = y_p[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore the boundary
    txx_copy = restore_boundaries(txx_copy, txx_bd, NPML, N, multiple=True)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd, NPML, N, multiple=True)
    txz_copy = restore_boundaries(txz_copy, txz_bd, NPML, N, multiple=True)
    p_copy = restore_boundaries(p_copy, p_bd, NPML, N, multiple=True)

    # The rest of your computation code...
    txx_x = diff_using_roll(y_txx, 2, False)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1)
    txz_x = diff_using_roll(y_txz, 2)

    p_x = diff_using_roll(y_p, 2, False)
    p_z = diff_using_roll(y_p, 1)

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region] = y_vx[(slice(None), slice(top,-N), slice(N,-N))]
    vz_copy[update_region] = y_vz[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd, NPML, N, multiple=True)
    vz_copy = restore_boundaries(vz_copy, vz_bd, NPML, N, multiple=True)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var = src_func(source_var, src_values, -1)

    return p_copy, vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy
    
def _time_step(*args, **kwargs):

    vp, vs, rho = args[:3]
    p, vx, vz, txx, tzz, txz = args[3:-3]
    dt, h, d = args[-3:]

    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    c = 0.5*dt*d

    txx_x = diff_using_roll(txx, 2, False)
    txz_z = diff_using_roll(txz, 1, False)
    tzz_z = diff_using_roll(tzz, 1)
    txz_x = diff_using_roll(txz, 2)

    p_x = diff_using_roll(p, 2, False)
    p_z = diff_using_roll(p, 1)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    """Calculate partial derivative of velocity components"""
    vx_x = diff_using_roll(y_vx, 2)
    vz_z = diff_using_roll(y_vz, 1, False)
    vx_z = diff_using_roll(y_vx, 1)
    vz_x = diff_using_roll(y_vz, 2, False)

    y_txx = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vx_x-vz_z)+(1-c)*txx)

    y_tzz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_z-vx_x)+(1-c)*tzz)

    y_txz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    y_p = (1+c)**-1*(-dt*(lame_lambda+lame_mu)*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)


    return y_p, y_vx, y_vz, y_txx, y_tzz, y_txz