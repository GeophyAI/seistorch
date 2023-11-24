import torch
from .utils import diff_using_roll

def _time_step(*args):

    vp, rho, Q = args[0:3]
    vx, vz, p, r = args[3:7]
    dt, dh, b = args[7:10]

    omega = 10.0

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

def _time_step_backward(vx, vz, p, r, vp, rho, Q, omega, b=None, dt=1.0, dh=1.0):

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