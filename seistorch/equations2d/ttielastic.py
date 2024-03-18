from .utils import diff_using_roll

def _time_step(*args):

    #vp, vs, rho, epsilon, gamma, delta = args[0:6]
    c11, c13, c33, c15, c35, c55, rho = args[0:7]

    vx, vz, txx, tzz, txz = args[7:12]
    dt, h, d = args[12:15]
    
    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    c = 0.5*dt*d

    # Equation A-8
    y_txx  = (1+c)**-1*(dt*h.pow(-1)*(c11*vx_x+c13*vz_z+c15*(vz_x+vx_z))+(1-c)*txx)
    # Equation A-9
    y_tzz  = (1+c)**-1*(dt*h.pow(-1)*(c33*vz_z+c13*vx_x+c35*(vz_x+vx_z))+(1-c)*tzz)

    # Equation A-10
    y_txz = (1+c)**-1*(dt*h.pow(-1)*(c15*vx_x+c35*vz_z+c55*vz_x+c55*vx_z)+(1-c)*txz)

    txx_x = diff_using_roll(y_txx, 2)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1, False)
    txz_x = diff_using_roll(y_txz, 2, False)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz


def _time_step_backward(*args):
    y_vx, y_vz, y_txx, y_tzz, y_txz = None, None, None, None, None
    pass

    return y_vx, y_vz, y_txx, y_tzz, y_txz