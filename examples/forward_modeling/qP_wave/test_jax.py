import numpy as np
import jax.numpy as jnp
import sys
import matplotlib.pyplot as plt
sys.path.append('../../..')
from seistorch.equations2d_jax.acoustic_vti_pml import _time_step
from seistorch.pml import generate_pml_coefficients_2d

pmln = 50
vp = np.load('./velocity_model/vp.npy')
delta_a = np.load('./velocity_model/delta_b.npy')
eps_a = np.load('./velocity_model/epsilon_b.npy')

vp = np.pad(vp, ((pmln, pmln), (pmln, pmln)), mode='edge')
eps_a = np.pad(eps_a, ((pmln, pmln), (pmln, pmln)), mode='edge')
delta_a = np.pad(delta_a, ((pmln, pmln), (pmln, pmln)), mode='edge')
pmlc = generate_pml_coefficients_2d(vp.shape)
pmlc = jnp.array(pmlc, dtype=jnp.float32)

domain = vp.shape
nz, nx = domain
p1 = jnp.zeros((1,*domain), dtype=jnp.float32)
p2 = jnp.zeros((1,*domain), dtype=jnp.float32)

src = [int(nz/2), int(nx/2)]

dt = 0.001
fm = 20
h = 10
nt = 2001
ricker = lambda t: (1-2*(np.pi*fm*(t-0.25))**2)*np.exp(-(np.pi*fm*(t-0.25))**2)

wave = ricker(np.arange(0, nt*dt, dt))

# show wave
plt.plot(np.arange(0, nt*dt, dt), wave)
plt.show()

for i in range(nt):

    p1, p2 = _time_step(vp, eps_a, delta_a, p1, p2, dt, h, pmlc)

    p1 = p1.at[0, src[0], src[1]].set(wave[i])

    if i % 100 == 0:
        plt.imshow(p1[0], cmap='seismic', aspect='auto')
        plt.colorbar()
        plt.show()