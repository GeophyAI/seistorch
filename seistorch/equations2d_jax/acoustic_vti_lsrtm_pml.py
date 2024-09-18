import jax.numpy as jnp
from .utils import laplace_with_kernel as lwk
from .utils import batch_convolve2d
from .kernels import kernelx, kernely

def gradient(y, h, kernel):
    operator = (2*h) ** (-1) * kernel
    return batch_convolve2d(y, operator)

def _time_step(*args, **kwargs):

    vp, eps, delta, m = args[0:4]
    p1, p2, sp1, sp2 = args[4:8]
    dt, h, b = args[8:11]

    # Method2: solver in frequency domain
    # 10.1190/geo2022-0292.1 EQ(12)
    shape = p1.shape[-2:]
    kx = jnp.fft.fftfreq(shape[0], d=h)
    kz = jnp.fft.fftfreq(shape[1], d=h)
    k_x, k_y = jnp.meshgrid(kx, kz, indexing='ij')
    numerator = -2*(eps-delta)*k_x**2*k_y**2
    denominator = (1+2*eps)*k_x**4+k_y**4+2*(1+delta)*k_x**2*k_y**2
    sk = numerator*((denominator+1e-18)**-1)
    sk = sk.real

    vp2dt2 = vp**2*dt**2
    a1 = 1+b*dt
    a2 = 1-b*dt

    # Background wavefield
    nabla_x = lwk(p1, h, kernelx)
    nabla_z = lwk(p1, h, kernely)
    # 10.1190/geo2022-0292.1 EQ(22)
    ani_laplace_p0 = vp2dt2*((1+2*eps)+sk)*nabla_x + vp2dt2*(1+sk)*nabla_z
    # pnext = 2*p1-p2 + ani_laplace_p0
    pnext = 2*a1**-1*p1 - a2*a1**(-1)*p2 + a1**-1*ani_laplace_p0
    # pnext = 2*a1**-1*p1 - a2*a1**(-1)*p2 + a1**-1*ani_laplace_p0

    # Scatter wavefield
    nabla_x_s = lwk(sp1, h, kernelx)
    nabla_z_s = lwk(sp1, h, kernely)
    ani_laplace_sp0 = vp2dt2*((1+2*eps)+sk)*nabla_x_s + vp2dt2*(1+sk)*nabla_z_s
    spnext = 2*a1**-1*sp1 - a2*a1**-1*sp2 + a1**-1*ani_laplace_sp0 + m*ani_laplace_p0
    # spnext = 2*sp1-sp2 + vp2dt2*((1+2*eps)+sk)*nabla_x_s + vp2dt2*(1+sk)*nabla_z_s+m*ani_laplace_p0

    return pnext, p1, spnext, sp1