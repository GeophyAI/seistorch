import numpy as np
import torch

def _corners(domain_shape, abs_N, d, dx, dy, multiple=False):
    Nx, Ny = domain_shape
    for j in range(Ny):
        for i in range(Nx):
            # Left-Top
            if not multiple:
                if i < abs_N+1 and j< abs_N+1:
                    if i < j: d[i,j] = dy[i,j]
                    else: d[i,j] = dx[i,j]
            # Left-Bottom
            if i > (Nx-abs_N-2) and j < abs_N+1:
                if i + j < Nx: d[i,j] = dx[i,j]
                else: d[i,j] = dy[i,j]
            # Right-Bottom
            if i > (Nx-abs_N-2) and j > (Ny-abs_N-2):
                if i - j > Nx-Ny: d[i,j] = dy[i,j]
                else: d[i,j] = dx[i,j]
            # Right-Top
            if not multiple:
                if i < abs_N+1 and j> (Ny-abs_N-2):
                    if i + j < Ny: d[i,j] = dy[i,j]
                    else: d[i,j] = dx[i,j]

    return d

def generate_pml_coefficients_2d(domain_shape, N=50, B=100., multiple=False):
    Nx, Ny = domain_shape

    # R = 10**(-((np.log10(N)-1)/np.log10(2))-3)
    #d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
    R = 1e-6; order = 2; cp = 1500.# Mao shibo Master
    d0 = (1.5*cp/N)*np.log10(R**-1)
    d_vals = d0 * torch.linspace(0.0, 1.0, N + 1) ** order
    d_vals = torch.flip(d_vals, [0])

    d_x = torch.zeros(Ny, Nx)
    d_y = torch.zeros(Ny, Nx)
    
    if N > 0:
        d_x[0:N + 1, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
        d_x[(Ny - N - 1):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
        if not multiple:
            d_y[:, 0:N + 1] = d_vals.repeat(Ny, 1)
        d_y[:, (Nx - N - 1):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

    _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)
    # _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, multiple)

    return _d


# def generate_pml_coefficients_2d(domain_shape, N=50, B=100., multiple=False):
#     Nx, Ny = domain_shape

#     # R = 10**(-((np.log10(N)-1)/np.log10(2))-3)
#     #d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
#     R = 1e-6; order = 2; cp = 1000.# Mao shibo Master
#     d0 = (1.5*cp/N)*np.log10(R**-1)
#     d_vals = d0 * torch.linspace(0.0, 1.0, N) ** order
#     d_vals = torch.flip(d_vals, [0])

#     d_x = torch.zeros(Ny, Nx)
#     d_y = torch.zeros(Ny, Nx)
    
#     if N > 0:
#         d_x[0:N, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
#         d_x[(Ny - N):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
#         if not multiple:
#             d_y[:, 0:N] = d_vals.repeat(Ny, 1)
#         d_y[:, (Nx - N):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

#     _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)
#     # _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, multiple)

#     return _d


# # A hybrid ABC for the 2D acoustic wave equation
# def generate_pml_coefficients_2d(domain_shape, N=50, B=100., multiple=False):

#     Nx, Ny = domain_shape

#     # Wang Weihong
#     d_vals = torch.linspace(0.0, N, N)/N
#     d_vals = torch.flip(d_vals, [0])

#     d_x = torch.zeros(Ny, Nx)
#     d_y = torch.zeros(Ny, Nx)
    
#     if N > 0:
#         d_x[0:N, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
#         d_x[(Ny - N):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
#         if not multiple:
#             d_y[:, 0:N] = d_vals.repeat(Ny, 1)
#         d_y[:, (Nx - N):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

#     _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)
#     # _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, multiple)

#     return _d

# def generate_pml_coefficients_2d(domain_shape, N=50, B=100., multiple=False):

#     Nx, Ny = domain_shape
#     B = 300.
#     # Wang Weihong
#     d_vals = B*(1-torch.cos(torch.linspace(0.0, N, N)*np.pi/(2*N)))
#     d_vals = torch.flip(d_vals, [0])

#     d_x = torch.zeros(Ny, Nx)
#     d_y = torch.zeros(Ny, Nx)
    
#     if N > 0:
#         d_x[0:N, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
#         d_x[(Ny - N):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
#         if not multiple:
#             d_y[:, 0:N] = d_vals.repeat(Ny, 1)
#         d_y[:, (Nx - N):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

#     _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)
#     # _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, multiple)

#     return _d

def generate_pml_coefficients_3d(domain_shape, N=50, B=100., multiple=False):
    nz, ny, nx = domain_shape
    # Cosine coefficients for pml
    idx = (torch.ones(N + 1) * (N+1)  - torch.linspace(0.0, (N+1), N + 1))/(2*(N+1))
    b_vals = torch.cos(torch.pi*idx)
    b_vals = torch.ones_like(b_vals) * B * (torch.ones_like(b_vals) - b_vals)

    b_x = torch.zeros((nz, ny, nx))
    b_y = torch.zeros((nz, ny, nx))
    b_z = torch.zeros((nz, ny, nx))

    b_x[:,0:N+1,:] = b_vals.repeat(nx, 1).transpose(0, 1)
    b_x[:,(ny - N - 1):ny,:] = torch.flip(b_vals, [0]).repeat(nx, 1).transpose(0, 1)

    b_y[:,:,0:N + 1] = b_vals.repeat(ny, 1)
    b_y[:,:,(nx - N - 1):nx] = torch.flip(b_vals, [0]).repeat(ny, 1)

    b_z[0:N + 1, :, :] = b_vals.view(-1, 1, 1).repeat(1, ny, nx)
    b_z[(nz - N - 1):nz + 1, :, :] = torch.flip(b_vals, [0]).view(-1, 1, 1).repeat(1, ny, nx)

    return torch.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2)



