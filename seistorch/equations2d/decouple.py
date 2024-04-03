import torch
from .utils import diff_using_roll, restore_boundaries

NPML = 49
N = 1
# Not finished yet
def _time_step(*args, **kwargs):
    vp, vs, rho = args[0:3]
    vxp, vxs, vzp, vzs, txxp, txxs, tzzp, tzzs, txzs = args[3:-3]
    dt, h, d = args[-3:]
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    vxp_x = diff_using_roll(vxp, 2, False)
    vzp_z = diff_using_roll(vzp, 1)
    vxp_z = diff_using_roll(vxp, 1, False)
    vzp_x = diff_using_roll(vzp, 2)

    vxs_x = diff_using_roll(vxs, 2, False)
    vzs_z = diff_using_roll(vzs, 1)
    vxs_z = diff_using_roll(vxs, 1, False)
    vzs_x = diff_using_roll(vzs, 2)

    c = 0.#0.5*dt*d
    y_txxp  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*(vxp_x+vxs_x+vzp_z+vzs_z)+(1-c)*txxp))
    y_tzzp  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*(vxp_x+vxs_x+vzp_z+vzs_z))+(1-c)*tzzp)
    # Equation A-10
    y_txxs = (1+c)**-1*(-dt*2*lame_mu*h.pow(-1)*(vzp_z+vzs_z)+(1-c)*txzs)
    y_tzzs = (1+c)**-1*(-dt*2*lame_mu*h.pow(-1)*(vxp_x+vxs_x)+(1-c)*tzzs)

    y_txzs = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vxp_z+vxs_z+vzp_x+vzs_x)+(1-c)*txzs)

    txxp_x = diff_using_roll(y_txxp, 2)
    tzzp_z = diff_using_roll(y_tzzp, 1, False)
    txxs_x = diff_using_roll(y_txxs, 2, False)
    tzzs_z = diff_using_roll(y_tzzs, 1, False)
    txzs_x = diff_using_roll(y_txzs, 2, False)
    txzs_z = diff_using_roll(y_txzs, 1, False)

    # Update y_vx
    y_vxp = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txxp_x)+(1-c)*vxp)
    y_vzp = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(tzzp_z)+(1-c)*vzp)
    # Update y_vz
    y_vxs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txxs_x+txzs_z)+(1-c)*vzs)
    y_vzs = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(tzzs_z+txzs_x)+(1-c)*vxs)

    return y_vxp, y_vxs, y_vzp, y_vzs, y_txxp, y_txxs, y_tzzp, y_tzzs, y_txzs

def _time_step_backward(*args, **kwargs):
    pass
    return

def _time_step_backward_multiple(*args, **kwargs):
    pass
    return