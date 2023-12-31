import torch
from torch.nn.functional import conv2d
from .utils import restore_boundaries

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
    
    operator = h ** (-2) * kernel#.to(y.device)
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=padding).squeeze(1)

def _time_step(*args):

    c = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]

    # b = 0
    # When b=0, without boundary conditon.
    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2 - b * dt**-1), h2)
                + torch.mul(c.pow(2), _laplacian(h1, h)))
                )

    return y, h1

def _time_step_backward(*args):

    vp = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    h_bd, _ = args[-2]
    src_type, src_func, src_values = args[-1]
    
    vp = vp.unsqueeze(0)
    b = b.unsqueeze(0)

    # b = 0

    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2 - b * dt**-1), h2)
                + torch.mul(vp.pow(2), _laplacian(h1, h)))
                )
    
    with torch.no_grad():
        y = restore_boundaries(y, h_bd)
    
    y = src_func(y, src_values, 1)

    return y, h1