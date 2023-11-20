from .utils import restore_boundaries, diff_using_roll

NPML = 49
N = 1

def _time_step(*args):

    vp, rho = args[0:2]
    vx, vz, p = args[2:5]
    dt, h, b = args[5:8]
    
    # Update velocity
    c = 0.5*dt*b

    p_x = diff_using_roll(p, 2, False)
    p_z = diff_using_roll(p, 1, False)

    y_vx = (1+c)**-1*(dt * rho.pow(-1)* p_x / h + (1-c)*vx)
    y_vz = (1+c)**-1*(dt * rho.pow(-1)* p_z / h + (1-c)*vz)

    # x -- 2
    # z -- 1
    vx_x = diff_using_roll(y_vx, 2)
    vz_z = diff_using_roll(y_vz, 1)

    y_p = (1+c)**-1*(vp**2*dt*rho*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)

    return y_vx, y_vz, y_p

def _time_step_backward(*args):

    vp, rho = args[0:2]
    vx, vz, p = args[2:5]
    dt, h, d = args[5:8]
    vx_bd, vz_bd, p_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))
    
    # Create a copy of the original tensors
    p_copy = p.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    vp = vp[compute_region_slice]
    rho = rho[compute_region_slice]
    p = p_copy[compute_region_slice]
    d = d[compute_region_slice]

    c = 0.5*dt*d

    # Update stress
    # x -- 2
    # z -- 1
    vx_x = diff_using_roll(vx, 2)[compute_region_slice]
    vz_z = diff_using_roll(vz, 1)[compute_region_slice]

    y_p = (1+c)**-1*(-vp**2*dt*rho*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)

    # Write back the results to the original tensors, but only within the update region
    p_copy[update_region_slice] = y_p[(slice(None), slice(N,-N), slice(N,-N))]
    # Restore the boundary
    p_copy = restore_boundaries(p_copy, p_bd, NPML, N)

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]

    # Update velocity
    p_x = diff_using_roll(p_copy, 2, False)[compute_region_slice]
    p_z = diff_using_roll(p_copy, 1, False)[compute_region_slice]

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*p_x+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*p_z+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]
    
    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd)
    vz_copy = restore_boundaries(vz_copy, vz_bd)
    
    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var = src_func(source_var, src_values, -1)

    return vx_copy, vz_copy, p_copy