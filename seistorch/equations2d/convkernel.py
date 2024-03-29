import torch

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

def generate_convolution_kernel(spatial_order, direction='all', nocenter=False):
    """Generate convolution kernel

    Args:
        n (int): The order of the taylor expansion

    Returns:
        _type_: Tensor, the convolution kernel
    """
    assert direction in ['all', 'x', 'y'], "direction must be one of ['all', 'x', 'y']"

    constant = even_intergrid(spatial_order)
    kernel_size = spatial_order + 1
    kernel = torch.zeros((kernel_size, kernel_size))
    center = spatial_order // 2

    if direction == 'all':
        kernel[center, center+1:] = constant
        kernel[center, 0:center] = constant.flip(0)

        kernel[center+1:, center] = constant
        kernel[0:center, center] = constant.flip(0)
        if not nocenter:
            kernel[center, center] = -2*2*torch.sum(constant)


    if direction == 'x':
        kernel[center, center+1:] = constant
        kernel[center, 0:center] = constant.flip(0)

    if direction == 'y':
        kernel[center+1:, center] = constant
        kernel[0:center, center] = constant.flip(0)

    if direction in ['x', 'y'] and not nocenter:
        kernel[center, center] = -2*torch.sum(constant)

    return kernel


spatial_order = 2
device = "cuda"
kernel = generate_convolution_kernel(spatial_order).unsqueeze(0).unsqueeze(0).to(device)
kernelx = generate_convolution_kernel(spatial_order, direction='x').unsqueeze(0).unsqueeze(0).to(device)
kernely = generate_convolution_kernel(spatial_order, direction='y').unsqueeze(0).unsqueeze(0).to(device)
# kernelx_nc = generate_convolution_kernel(spatial_order, direction='x', nocenter=True).unsqueeze(0).unsqueeze(0).to(device)
# kernely_nc = generate_convolution_kernel(spatial_order, direction='y', nocenter=True).unsqueeze(0).unsqueeze(0).to(device)
kernelx_nc = torch.tensor([[[[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]]).to(device)
kernely_nc = torch.tensor([[[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]).to(device)

padding = kernel.shape[-1]//2
