from .utils import diff_using_roll, restore_boundaries
import torch
NPML = 49
N = 1

def _time_step(*args, **kwargs):

    vp, vs, rho = args[:3]
    theta, omega, vpx, vpz, vsx, vsz, _, _ = args[3:-3]
    dt, h, d = args[-3:]

    # lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    # lame_mu = rho*(vs.pow(2))

    c = 0.5*dt*d
    theta_x = diff_using_roll(theta, 2)
    theta_z = diff_using_roll(theta, 1)
    omega_x = diff_using_roll(omega, 2, False)
    omega_z = diff_using_roll(omega, 1, False)

    y_vpx = (1+c)**-1*(vp**2*dt*h**-1*theta_x+(1-c)*vpx)
    y_vpz = (1+c)**-1*(vp**2*dt*h**-1*theta_z+(1-c)*vpz)
    y_vsx = (1+c)**-1*(-vs**2*dt*h**-1*omega_z+(1-c)*vsx)
    y_vsz = (1+c)**-1*(vs**2*dt*h**-1*omega_x+(1-c)*vsz)

    vx = y_vpx+y_vsx
    vz = y_vpz+y_vsz

    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    y_theta = (1+c)**-1*(dt*h.pow(-1)*(vx_x+vz_z)+(1-c)*theta)
    y_omega = (1+c)**-1*(dt*h.pow(-1)*(vz_x-vx_z)+(1-c)*omega)

    return y_theta, y_omega, y_vpx, y_vpz, y_vsx, y_vsz, vx, vz

def _time_step_backward(*args, **kwargs):

    vp, vs, rho = args[0:3]
    theta, omega, vpx, vpz, vsx, vsz, _, _ = args[3:11]
    dt, h, d = args[11:14]
    theta_bd, omega_bd, vpx_bd, vpz_bd, vsx_bd, vsz_bd, _, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    vp = vp.unsqueeze(0)[compute_region_slice]
    vs = vs.unsqueeze(0)[compute_region_slice]
    rho = rho.unsqueeze(0)[compute_region_slice]
    d = d.unsqueeze(0)[compute_region_slice]

    c = 0.5*dt*d

    vx = vpx+vsx
    vz = vpz+vsz

    theta_copy, omega_copy = theta.clone(), omega.clone()

    theta = theta[compute_region_slice]
    omega = omega[compute_region_slice]
    vx = vx[compute_region_slice]
    vz = vz[compute_region_slice]

    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    y_theta = (1+c)**-1*(-dt*h.pow(-1)*(vx_x+vz_z)+(1-c)*theta)
    y_omega = (1+c)**-1*(-dt*h.pow(-1)*(vx_z-vz_x)+(1-c)*omega)


    theta_copy[update_region_slice] = y_theta[(slice(None), slice(N,-N), slice(N,-N))]
    omega_copy[update_region_slice] = y_omega[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore boundaries
    theta_copy = restore_boundaries(theta_copy, theta_bd)
    omega_copy = restore_boundaries(omega_copy, omega_bd)

    vpx_copy, vpz_copy, vsx_copy, vsz_copy = vpx.clone(), vpz.clone(), vsx.clone(), vsz.clone()

    vpx = vpx_copy[compute_region_slice]
    vpz = vpz_copy[compute_region_slice]
    vsx = vsx_copy[compute_region_slice]
    vsz = vsz_copy[compute_region_slice]

    theta_x = diff_using_roll(y_theta, 2)
    theta_z = diff_using_roll(y_theta, 1)
    omega_x = diff_using_roll(y_omega, 2, False)
    omega_z = diff_using_roll(y_omega, 1, False)
    
    y_vpx = (1+c)**-1*(-vp**2*dt*h**-1*theta_x+(1-c)*vpx)
    y_vpz = (1+c)**-1*(-vp**2*dt*h**-1*theta_z+(1-c)*vpz)
    y_vsx = (1+c)**-1*(-vs**2*dt*h**-1*omega_z+(1-c)*vsx)
    y_vsz = (1+c)**-1*(+vs**2*dt*h**-1*omega_x+(1-c)*vsz)

    vpx_copy[update_region_slice] = y_vpx[(slice(None), slice(N,-N), slice(N,-N))]
    vpz_copy[update_region_slice] = y_vpz[(slice(None), slice(N,-N), slice(N,-N))]
    vsx_copy[update_region_slice] = y_vsx[(slice(None), slice(N,-N), slice(N,-N))]
    vsz_copy[update_region_slice] = y_vsz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore boundaries
    vpx_copy = restore_boundaries(vpx_copy, vpx_bd)
    vpz_copy = restore_boundaries(vpz_copy, vpz_bd)
    vsx_copy = restore_boundaries(vsx_copy, vsx_bd)
    vsz_copy = restore_boundaries(vsz_copy, vsz_bd)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var.data.copy_(src_func(source_var, src_values, -1))
    
    return theta_copy, omega_copy, vpx_copy, vpz_copy, vsx_copy, vsz_copy, vx, vz

def _time_step_backward_multiple(*args, **kwargs):
    top=0

    vp, vs, rho = args[0:3]
    theta, omega, vpx, vpz, vsx, vsz, _, _ = args[3:11]
    dt, h, d = args[11:14]
    theta_bd, omega_bd, vpx_bd, vpz_bd, vsx_bd, vsz_bd, _, _ = args[-2]
    src_type, src_func, src_values = args[-1]

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(top, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(top, -NPML-N), slice(NPML+N, -NPML-N))

    vp = vp.unsqueeze(0)[compute_region_slice]
    vs = vs.unsqueeze(0)[compute_region_slice]
    rho = rho.unsqueeze(0)[compute_region_slice]
    d = d.unsqueeze(0)[compute_region_slice]

    c = 0.5*dt*d

    vx = vpx+vsx
    vz = vpz+vsz

    theta_copy, omega_copy = theta.clone(), omega.clone()

    theta = theta[compute_region_slice]
    omega = omega[compute_region_slice]
    vx = vx[compute_region_slice]
    vz = vz[compute_region_slice]

    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    y_theta = (1+c)**-1*(-dt*h.pow(-1)*(vx_x+vz_z)+(1-c)*theta)
    y_omega = (1+c)**-1*(-dt*h.pow(-1)*(vx_z-vz_x)+(1-c)*omega)


    theta_copy[update_region_slice] = y_theta[(slice(None), slice(top,-N), slice(N,-N))]
    omega_copy[update_region_slice] = y_omega[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore boundaries
    theta_copy = restore_boundaries(theta_copy, theta_bd, multiple=True)
    omega_copy = restore_boundaries(omega_copy, omega_bd, multiple=True)

    vpx_copy, vpz_copy, vsx_copy, vsz_copy = vpx.clone(), vpz.clone(), vsx.clone(), vsz.clone()

    vpx = vpx_copy[compute_region_slice]
    vpz = vpz_copy[compute_region_slice]
    vsx = vsx_copy[compute_region_slice]
    vsz = vsz_copy[compute_region_slice]

    theta_x = diff_using_roll(y_theta, 2)
    theta_z = diff_using_roll(y_theta, 1)
    omega_x = diff_using_roll(y_omega, 2, False)
    omega_z = diff_using_roll(y_omega, 1, False)
    
    y_vpx = (1+c)**-1*(-vp**2*dt*h**-1*theta_x+(1-c)*vpx)
    y_vpz = (1+c)**-1*(-vp**2*dt*h**-1*theta_z+(1-c)*vpz)
    y_vsx = (1+c)**-1*(-vs**2*dt*h**-1*omega_z+(1-c)*vsx)
    y_vsz = (1+c)**-1*(+vs**2*dt*h**-1*omega_x+(1-c)*vsz)

    vpx_copy[update_region_slice] = y_vpx[(slice(None), slice(top,-N), slice(N,-N))]
    vpz_copy[update_region_slice] = y_vpz[(slice(None), slice(top,-N), slice(N,-N))]
    vsx_copy[update_region_slice] = y_vsx[(slice(None), slice(top,-N), slice(N,-N))]
    vsz_copy[update_region_slice] = y_vsz[(slice(None), slice(top,-N), slice(N,-N))]

    # Restore boundaries
    vpx_copy = restore_boundaries(vpx_copy, vpx_bd, multiple=True)
    vpz_copy = restore_boundaries(vpz_copy, vpz_bd, multiple=True)
    vsx_copy = restore_boundaries(vsx_copy, vsx_bd, multiple=True)
    vsz_copy = restore_boundaries(vsz_copy, vsz_bd, multiple=True)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        source_var.data.copy_(src_func(source_var, src_values, -1))
    
    return theta_copy, omega_copy, vpx_copy, vpz_copy, vsx_copy, vsz_copy, vx, vz
