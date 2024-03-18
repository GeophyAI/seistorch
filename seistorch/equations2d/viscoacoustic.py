import torch
from .utils import diff_using_roll

NPML = 49
N = 1

def _time_step(*args, **kwargs):

    vp, rho, Q = args[0:3]
    vx, vz, p, r = args[3:7]
    dt, dh, b = args[7:10]

    omega = 4.0

    Kappa = rho*vp**2
    t_sigma = omega**-1*(torch.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1
    t = t_epslion/(t_sigma-1e-8) - 1

    # x -- 2
    # z -- 1
    yvx = vx - rho**-1 * dt * dh**-1 * diff_using_roll(p, 2, False) - dt * b * vx
    yvz = vz - rho**-1 * dt * dh**-1 * diff_using_roll(p, 1, False) - dt * b * vz
    nabla = dh ** -1 * (diff_using_roll(yvx, dim=2)+diff_using_roll(yvz, dim=1))

    yr = r - dt * t_sigma**-1 * (Kappa*t*nabla + r) - dt * b * r

    yp = p - dt * Kappa * (t+1) * nabla - 0.5* dt * (yr+r) - dt * b * p

    return yvx, yvz, yp, yr

def _time_step_backward_multiple(*args, **kwargs):

    top = 0
    vp, rho, Q = args[0:3]
    vx, vz, p, r = args[3:7]
    dt, dh, b = args[7:10]
    vx_bd, vz_bd, p_bd, r_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    rho = rho.unsqueeze(0)
    Q = Q.unsqueeze(0)
    d = d.unsqueeze(0)

    omega = 4.0

    Kappa = rho*vp**2
    t_sigma = omega**-1*(torch.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1
    t = t_epslion/(t_sigma-1e-8) - 1

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(top, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(top, -NPML-N), slice(NPML+N, -NPML-N))
    
    p_copy = p.clone()

    vp = vp[compute_region_slice]
    rho = rho[compute_region_slice]
    Q = Q[compute_region_slice]
    p = p_copy[compute_region_slice]
    d = d[compute_region_slice]


    # x -- 2
    # z -- 1
    vx_x = diff_using_roll(yvx, dim=2)[compute_region_slice]
    vz_z = diff_using_roll(yvz, dim=1)[compute_region_slice]

    y_p = p + dt*dh**-1*Kappa*(t+1)* (vx_x+vz_z) + 0.5*dt*(yr+r) +dt*b*p
    y_r = r + dt*dh**-1*t_sigma**-1*Kappa*t

    yvx = vx - rho**-1 * dt * dh**-1 * diff_using_roll(p, 2, False) - dt * b * vx
    yvz = vz - rho**-1 * dt * dh**-1 * diff_using_roll(p, 1, False) - dt * b * vz

    yr = r - dt * t_sigma**-1 * (Kappa*t*nabla + r) - dt * b * r

    yp = p - dt * Kappa * (t+1) * nabla - 0.5* dt * (yr+r) - dt * b * p

    return yvx, yvz, yp, yr
