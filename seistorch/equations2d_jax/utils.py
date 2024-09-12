from jax import vmap
import jax.numpy as jnp
from jax.scipy.signal import convolve2d as conv2d

def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

# batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))
batch_convolve2d = vmap(_laplace, in_axes=(0, None))

def laplace(u, h):
    kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 3x3 kernel
    return batch_convolve2d(u, kernel) / (h ** 2)


def diff_using_roll(input, axis=-1, forward=True, padding_value=0):

    def forward_diff(x, axis=-1, padding_value=0):
        """
        Compute the forward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            axis (int, optional): The axis along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The forward difference of the input tensor.
        """
        # x[:,0] = padding_value
        diff = x - jnp.roll(x, shift=1, axis=axis)
        if axis == 1:
            diff = diff.at[:, 0].set(padding_value)
        elif axis == 2:
            diff = diff.at[..., 0].set(padding_value)  # pad with specified value
        return diff

    def backward_diff(x, axis=-1, padding_value=0):
        """
        Compute the backward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The backward difference of the input tensor.
        """
        # x[...,-1] = padding_value
        diff = jnp.roll(x, shift=-1, axis=axis) - x
        if axis == 1:
            diff = diff.at[:, -1].set(padding_value)
        elif axis == 2:
            diff = diff.at[..., -1].set(padding_value)  # pad with specified value
        return diff

    if forward:
        return forward_diff(input, axis=axis)
    else:
        return backward_diff(input, axis=axis)