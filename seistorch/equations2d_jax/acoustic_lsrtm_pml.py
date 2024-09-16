from .utils import laplace

def _time_step(*args, **kwargs):
    
    vp, m = args[0:2]
    h1, h2, sh1, sh2 = args[2:6]
    dt, h, b = args[6:9]

    a = (dt**-2 + b * dt**-1)**(-1)

    # y = a*(2. / dt**2 * h1 - (dt**-2-b*dt**-1)*h2 + c**2*_laplacian(h1, h))

    # background wavefield
    vp2_nabla_p0 = vp**2*laplace(h1, h)
    # p0 = 2*h1-h2 + vp2_nabla_p0*dt**2 # No boundary condition
    p0 = a*(2. / dt**2 * h1 - (dt**-2-b*dt**-1)*h2 + vp2_nabla_p0)
    
    # scatter wavefield
    vp2_nabla_sh0 = vp**2*laplace(sh1, h)
    # sh0 = 2*sh1-sh2 + vp2_nabla_sh0*dt**2 + m*vp2_nabla_p0*dt**2 # No boundary condition
    sh0 = a*(2. / dt**2 * sh1 - (dt**-2-b*dt**-1)*sh2 + vp2_nabla_sh0 + m*vp2_nabla_p0)

    return p0, h1, sh0, sh1