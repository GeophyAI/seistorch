import torch
from torch.nn.functional import conv2d
from .utils import restore_boundaries
from .acoustic_habc import habc

def even_intergrid(n: int):
    """Calculate the coefficients of taylar expansion of even intergrid

    Args:
        n (int): The spatial order of the laplace operator.
    """
    def fact(n):
        if n == 0:
            return 1
        else:
            return n * fact(n-1)
    
    MAX = n // 2
    matrix = torch.zeros((MAX, MAX))
    constant_one = torch.zeros((MAX,))
    for i in range(MAX):
        for j in range(MAX):
            matrix[i, j] = (j + 1)**(2*(i+1) - 2)
        if i != ((2+1)//2 - 1):
            constant_one[i] = 0
        else:
            constant_one[i] = (1**(2*(i+1)-2) * fact(2*(i+1)-2))

    C = torch.linalg.solve(matrix, constant_one.unsqueeze(1))
    constant = C.squeeze() / torch.tensor([(i+1)**2 for i in range(MAX)], dtype=torch.float32)
    return constant

def generate_convolution_kernel(spatial_order):
    """Generate convolution kernel

    Args:
        n (int): The order of the taylor expansion

    Returns:
        _type_: Tensor, the convolution kernel
    """
    constant = even_intergrid(spatial_order)
    kernel_size = spatial_order + 1
    kernel = torch.zeros((kernel_size, kernel_size))
    center = spatial_order // 2
    kernel[center, center+1:] = constant
    kernel[center, 0:center] = constant.flip(0)

    kernel[center+1:, center] = constant
    kernel[0:center, center] = constant.flip(0)

    kernel[center, center] = -2*2*torch.sum(constant)

    return kernel

spatial_order = 2
device = "cuda"
kernel = generate_convolution_kernel(spatial_order).unsqueeze(0).unsqueeze(0).to(device)
padding = kernel.shape[-1]//2

def _laplacian(y, h):
    """Laplacian operator"""
    # kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(y.device)
    # kernel = torch.tensor([[[[0.0, 0.0, -0.083, 0.0, 0.0],
    #                          [0.0, 0.0, 1.333, 0.0, 0.0],
    #                          [-0.083, 1.333, -2.5, 1.333, -0.083],
    #                          [0.0, 0.0, 1.333, 0.0, 0.0],
    #                          [0.0, 0.0, -0.083, 0.0, 0.0]]]]).to(y.device)
    
    operator = h ** (-2) * kernel.to(y.device)
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=padding).squeeze(1)

def _time_step(*args, **kwargs):

    vp, Q = args[0], args[1]
    h1, h2 = args[2:4]
    dt, h, b = args[4:7]

    habc_masks = kwargs['habcs']
    
    #vp = vp.unsqueeze(0)

    omega = 16.5

    t_sigma = omega**-1*(torch.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1.
    t = t_epslion/(t_sigma-1e-8) - 1.

    # original wave equation
    y = 2*h1-h2 + vp**2*_laplacian(h1, h)*dt**2

    # phase move
    y -= ((1-torch.sqrt(Q**2+1))*Q**-2)*vp**2*_laplacian(h1, h)*dt**2

    # calculate wavenumber of h1
    dy = (h1 - h2)*dt**-1
    fft_dy = torch.fft.fft2(dy, dim=(-2, -1))
    shape = y.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(y.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(y.device)
    # shift to center
    # kx = torch.fft.fftshift(kx)
    # kz = torch.fft.fftshift(kz)
    k_x, k_y = torch.meshgrid(kx, kz, indexing='ij')
    k = torch.sqrt(k_x ** 2 + k_y ** 2)

    temp = torch.fft.ifft2(k*fft_dy, dim=(-2, -1)).real
    # amplitude decay
    # y -= (dt**2*t*vp/2)*temp

    # habc
    y = habc(y, h1, h2, vp, b, dt, h, maskidx = habc_masks)

    return y, h1

def _time_step_backward_multiple(*args, **kwargs):

    vp, Q = args[0], args[1]
    h1, h2 = args[2:4]
    dt, h, b = args[4:7]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    # vp = vp.unsqueeze(0)
    # b = b.unsqueeze(0)

    # b = 0

    omega = 5.0

    t_sigma = omega**-1*(torch.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1.
    t = t_epslion/(t_sigma-1e-8) - 1.

    y = 2*h1-h2 + vp**2*_laplacian(h1, h)*dt**2
    # phase 
    y -= ((1-torch.sqrt(Q**2+1))*Q**-2)*vp**2*_laplacian(h1, h)*dt**2

    # calculate wavenumber of h1
    dy = (h1 - h2)*dt**-1
    fft_dy = torch.fft.fft2(dy, dim=(-2, -1))
    shape = y.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(y.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(y.device)

    k_x, k_y = torch.meshgrid(kx, kz, indexing='ij')
    k = torch.sqrt(k_x ** 2 + k_y ** 2)

    temp = torch.fft.ifft2(k*fft_dy, dim=(-2, -1)).real
    # amplitude
    y -= (dt**2*t*vp/2)*temp

    with torch.no_grad():
        y = restore_boundaries(y, h_bd, multiple=True)
    
    y = src_func(y, src_values, 1)

    return y, h1


def _time_step_backward(*args, **kwargs):

    vp, Q = args[0], args[1]
    h1, h2 = args[2:4]
    dt, h, b = args[4:7]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    # vp = vp.unsqueeze(0)
    # b = b.unsqueeze(0)

    # b = 0

    omega = 5.0

    t_sigma = omega**-1*(torch.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1.
    t = t_epslion/(t_sigma-1e-8) - 1.

    y = 2*h1-h2 + vp**2*_laplacian(h1, h)*dt**2
    # phase 
    y -= ((1-torch.sqrt(Q**2+1))*Q**-2)*vp**2*_laplacian(h1, h)*dt**2

    # calculate wavenumber of h1
    dy = (h1 - h2)*dt**-1
    fft_dy = torch.fft.fft2(dy, dim=(-2, -1))
    shape = y.shape[-2:]
    kx = torch.fft.fftfreq(shape[0], d=h).to(y.device)
    kz = torch.fft.fftfreq(shape[1], d=h).to(y.device)

    k_x, k_y = torch.meshgrid(kx, kz, indexing='ij')
    k = torch.sqrt(k_x ** 2 + k_y ** 2)

    temp = torch.fft.ifft2(k*fft_dy, dim=(-2, -1)).real
    # amplitude
    y -= (dt**2*t*vp/2)*temp

    with torch.no_grad():
        y = restore_boundaries(y, h_bd)
    
    y = src_func(y, src_values, 1)

    return y, h1