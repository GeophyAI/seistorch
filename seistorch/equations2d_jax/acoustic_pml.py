from .utils import laplace

def _time_step(*args, **kwargs):

    vp = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    spatial_order = args[6]
    
    _laplace_u = laplace(h1, h, spatial_order)
    a = (dt**-2 + b * dt**-1)**(-1)
    # u_next = 2*u_now - u_pre + vp**2*dt**2*_laplace_u
    y = a*(2. / dt**2 * h1 - (dt**-2-b*dt**-1)*h2 + vp**2*_laplace_u)
    
    return y, h1