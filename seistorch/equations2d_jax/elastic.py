import torch
from .utils import diff_using_roll

def _time_step(*args, **kwargs):
    vp, vs, rho = args[0:3]
    vx, vz, txx, tzz, txz = args[3:8]
    dt, h, d = args[8:11]
    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*(vs**2)
    c = 0.5*dt*d

    vx_x = diff_using_roll(vx, 2)
    vz_z = diff_using_roll(vz, 1, False)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2, False)

    # Equation A-8
    y_txx  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    # Equation A-9
    y_tzz  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    # Equation A-10
    y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = diff_using_roll(y_txx, 2, False)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1)
    txz_x = diff_using_roll(y_txz, 2)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz