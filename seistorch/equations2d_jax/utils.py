from jax import vmap
import jax.numpy as jnp
from jax.scipy.signal import convolve2d as conv2d

def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

# batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))
batch_convolve2d = vmap(_laplace, in_axes=(0, None))

def laplace(u, h, order=2):
    kernel = generate_convolution_kernel(order)
    # kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 3x3 kernel
    return batch_convolve2d(u, kernel) / (h ** 2)

def laplace_with_kernel(u, h, kernel):
    return batch_convolve2d(u, kernel) / (h ** 2)

def staggered_grid_coes(M):
    # 2*M: difference order
    a = jnp.zeros(M, dtype=jnp.float32)
    
    for m in range(1, M + 1):
        a_m = (-1) ** (m + 1) / (2 * m - 1)
        
        prod = 1.0
        for n in range(1, M + 1):
            if n != m:
                numerator = (2 * n - 1) ** 2
                denominator = numerator - (2 * m - 1) ** 2
                prod *= jnp.abs(numerator / denominator)
        
        a_m *= prod
        
        a = a.at[m - 1].set(a_m)
    
    return a

def normal_grid_coes(M):
    # 2*M: difference order
    a_m = jnp.zeros(M)
    
    for m in range(1, M + 1):
        product = 1.0
        for n in range(1, M + 1):
            if n != m:
                product *= jnp.abs(n**2 / (n**2 - m**2))
        
        a_m = a_m.at[m - 1].set((-1)**(m + 1) / (m**2) * product)

    return a_m


def generate_convolution_kernel(spatial_order):
    """Generate convolution kernel

    Args:
        n (int): The order of the taylor expansion(Must be even)

    Returns:
        _type_: Tensor, the convolution kernel
    """

    constant = normal_grid_coes(spatial_order//2)
    kernel_size = spatial_order + 1
    kernel = jnp.zeros((kernel_size, kernel_size),dtype=jnp.float32)
    center = spatial_order // 2

    kernel = kernel.at[center, center+1:].set(constant)
    kernel = kernel.at[center, 0:center].set(constant[::-1])

    kernel = kernel.at[center+1:, center].set(constant)
    kernel = kernel.at[0:center, center].set(constant[::-1])

    kernel = kernel.at[center, center].set(-2*2*jnp.sum(constant))

    return kernel


# 2x faster than using jnp.roll
def diff_using_roll(input, axis=-1, forward=True, padding_value=0, order=2):
    M = order//2
    a = staggered_grid_coes(order)

    def forward_diff(x, axis, padding_value):
        """
        Compute the forward difference of an input tensor along a given dimension.
        """
        diff = jnp.zeros_like(x)
        pad_mask = jnp.zeros(x.shape, dtype=bool)
        # Use for loop
        for i in range(1, M+1):
            rolled_x = jnp.roll(x, shift=i, axis=axis)
            diff += a[i-1] * (x - rolled_x)
            pad_mask = pad_mask.at[tuple(slice(None) if d != axis else i-1 for d in range(x.ndim))].set(True)
        
        return jnp.where(pad_mask, padding_value, diff)

    def backward_diff(x, axis, padding_value):
        """
        Compute the backward difference of an input tensor along a given dimension.
        """
        diff = jnp.zeros_like(x)
        pad_mask = jnp.zeros(x.shape, dtype=bool)

        for i in range(1, M+1):
            rolled_x = jnp.roll(x, shift=-i, axis=axis)
            diff += a[i-1] * (rolled_x - x)
            pad_mask = pad_mask.at[tuple(slice(None) if d != axis else -(i) for d in range(x.ndim))].set(True)
        
        return jnp.where(pad_mask, padding_value, diff)

    if forward:
        return forward_diff(input, axis, padding_value)
    else:
        return backward_diff(input, axis, padding_value)