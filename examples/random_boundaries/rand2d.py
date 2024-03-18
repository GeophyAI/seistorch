import numpy as np
import matplotlib.pyplot as plt

import numpy as np

np.random.seed(20231219)

# 请替换这里的参数值
el = 10
l = 50
dx = 20.0
dz = 20.0
v_min = 1500.0
v_max = 5500


def random_fill_2d(v, el, l, dx, dz, v_min, v_max):

    v = np.pad(v, ((l, l), (l, l)), 'constant')
    nz, nx = v.shape
    px = np.random.rand(nz, nx)
    pz = np.random.rand(nz, nx)

    for i in range(nz):
        for j in range(nx):
            if not (i % el) and not (j % el) and not (i >= l and i < nz - l and j >= l and j < nx - l):
                v[i, j] = v_min + np.random.randint(v_max - v_min + 1) + np.random.rand()

    for i in range(nz):
        for j in range(nx):
            if (i % el or j % el) and not (i >= l and i < nz - l and j >= l and j < nx - l):
                if px[i, j] <= (i % el * dx) / (el * dx) and pz[i, j] <= (j % el * dz) / (el * dz):
                    if i - i % el + el < nz and j - j % el + el < nx:
                        v[i, j] = v[i - i % el + el, j - j % el + el]
                    elif i - i % el + el < nz and j - j % el + el >= nx:
                        v[i, j] = v[i - i % el + el, j - j % el]
                    elif i - i % el + el >= nz and j - j % el + el < nx:
                        v[i, j] = v[i - i % el, j - j % el + el]
                    else:
                        v[i, j] = v[i - i % el, j - j % el]

                elif px[i, j] <= (i % el * dx) / (el * dx) and pz[i, j] > (j % el * dz) / (el * dz):
                    if i - i % el + el < nz:
                        v[i, j] = v[i - i % el + el, j - j % el]
                    else:
                        v[i, j] = v[i - i % el, j - j % el]

                elif px[i, j] > (i % el * dx) / (el * dx) and pz[i, j] > (j % el * dz) / (el * dz):
                    v[i, j] = v[i - i % el, j - j % el]

                else:
                    if j - j % el + el < nx:
                        v[i, j] = v[i - i % el, j - j % el + el]
                    else:
                        v[i, j] = v[i - i % el, j - j % el]

    return v

vel_ori = np.load("/home/shaowinw/Desktop/seistorch/examples/models/marmousi_model/true_vp.npy")
# vel_ori = np.ones_like(vel_ori) * 2000
nz, nx= vel_ori.shape
vel_rd = random_fill_2d(vel_ori, el, l, dx, dz, v_min, v_max)
print(vel_rd.shape, vel_ori.shape)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(vel_ori, cmap="jet", vmin=v_min, vmax=v_max, aspect="auto")
axes[1].imshow(vel_rd, cmap="jet", vmin=v_min, vmax=v_max, aspect="auto")
# plt.colorbar(axes[0].imshow(vel_ori, cmap="jet"), ax=axes[0])
# plt.colorbar(axes[1].imshow(vel_rd, cmap="jet"), ax=axes[1])
plt.tight_layout()
plt.show()

plt.plot(vel_ori[:, 500], label="original")
plt.plot(vel_rd[:, 500], label="random")
plt.legend()
plt.show()

plt.plot(vel_ori[100], label="original")
plt.plot(vel_rd[100], label="random")
plt.legend()
plt.show()

# rd_bound_2d("your_input_model.bin", nz, nx, el, m, l, dx, dz, v_min, v_max, vmin)

