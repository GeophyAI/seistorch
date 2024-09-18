import jax.numpy as jnp
from .utils import laplace_with_kernel as lwk
from .utils import batch_convolve2d
from .kernels import kernelx, kernely, kernelx_nc, kernely_nc

def gradient(y, h, kernel):
    operator = (2*h) ** (-1) * kernel
    return batch_convolve2d(y, operator)

def _time_step(*args, **kwargs):

    vp, eps, delta, theta, m = args[0:5]
    p1, p2, sp1, sp2 = args[5:9]
    dt, h, b = args[9:12]
    habc_masks = kwargs['habcs']

    # from degree to radian
    theta = jnp.deg2rad(theta)
    sin0 = jnp.sin(theta)
    cos0 = jnp.cos(theta)
    sin20 = jnp.sin(2*theta)

    nabla_x = lwk(p1, h, kernelx)
    nabla_z = lwk(p1, h, kernely)

    dpdx = gradient(p1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz = gradient(dpdx, h, kernely_nc)

    # 10.1190/geo2022-0292.1 EQ(A-5)
    shape = p1.shape[-2:]
    kx = jnp.fft.fftfreq(shape[0], d=h)
    kz = jnp.fft.fftfreq(shape[1], d=h)
    k_x, k_z = jnp.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    denominator = (1+2*eps)*(k_x*cos0-k_z*sin0)**4+(k_x*sin0+k_z*cos0)**4+2*(1+delta)*(k_x*cos0-k_z*sin0)**2*(k_x*sin0+k_z*cos0)**2
    sd = numerator*((denominator+1e-18)**-1)
    sd = sd.real

    vp2dt2 = vp**2*dt**2

    # Background wavefield
    # 10.1190/geo2022-0292.1 EQ(A-7)
    ani_laplace_p0 = vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z-2*eps*vp2dt2*sin20*dpdxdz
    pnext = 2*p1-p2 + ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = lwk(sp1, h, kernelx)
    nabla_z_s = lwk(sp1, h, kernely)

    dpdx_s = gradient(sp1, h, kernelx_nc) # 10.1190/geo2022-0292.1 EQ(21)
    dpdxdz_s = gradient(dpdx_s, h, kernely_nc)
    spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)*cos0**2+sin0**2+sd)*nabla_x_s + vp2dt2*((1+2*eps)*sin0**2+cos0**2+sd)*nabla_z_s-2*eps*vp2dt2*sin20*dpdxdz_s+m*ani_laplace_p0

    return pnext, p1, spnext, sp1