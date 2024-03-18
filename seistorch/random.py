import numpy as np
import torch

# TODO:
# 1. 10.1190/GEO2014-0542.1 random boundary condition with random shape

# def random_fill_2d(v, el, l, dx, dz, v_min, v_max, usepad=False):

#     if usepad:
#         v = np.pad(v, ((l, l), (l, l)), 'constant')
        
#     nz, nx = v.shape
#     px = np.random.rand(nz, nx)
#     pz = np.random.rand(nz, nx)
#     dev = v.device
#     for i in range(nz):
#         for j in range(nx):
#             if not (i % el) and not (j % el) and not (i >= l and i < nz - l and j >= l and j < nx - l):
#                 #v[i, j] = v_min + np.random.randint(v_max - v_min + 1) + np.random.rand()
#                 v[i, j] = v_min + torch.randint(v_max - v_min + 1, (1,), device=dev) + torch.rand(1, device=dev)
#     for i in range(nz):
#         for j in range(nx):
#             if (i % el or j % el) and not (i >= l and i < nz - l and j >= l and j < nx - l):
#                 if px[i, j] <= (i % el * dx) / (el * dx) and pz[i, j] <= (j % el * dz) / (el * dz):
#                     if i - i % el + el < nz and j - j % el + el < nx:
#                         v[i, j] = v[i - i % el + el, j - j % el + el]
#                     elif i - i % el + el < nz and j - j % el + el >= nx:
#                         v[i, j] = v[i - i % el + el, j - j % el]
#                     elif i - i % el + el >= nz and j - j % el + el < nx:
#                         v[i, j] = v[i - i % el, j - j % el + el]
#                     else:
#                         v[i, j] = v[i - i % el, j - j % el]

#                 elif px[i, j] <= (i % el * dx) / (el * dx) and pz[i, j] > (j % el * dz) / (el * dz):
#                     if i - i % el + el < nz:
#                         v[i, j] = v[i - i % el + el, j - j % el]
#                     else:
#                         v[i, j] = v[i - i % el, j - j % el]

#                 elif px[i, j] > (i % el * dx) / (el * dx) and pz[i, j] > (j % el * dz) / (el * dz):
#                     v[i, j] = v[i - i % el, j - j % el]

#                 else:
#                     if j - j % el + el < nx:
#                         v[i, j] = v[i - i % el, j - j % el + el]
#                     else:
#                         v[i, j] = v[i - i % el, j - j % el]

#     return v


def random_fill_2d(v, el, l, dx, dz, v_min, v_max, usepad=False):

    if usepad:
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