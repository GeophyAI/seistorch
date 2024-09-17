from flax import linen as nn
import flax.linen as fnn
import jax.numpy as jnp
from jax.nn.initializers import uniform as uniform_init
from jax.nn.initializers import zeros
from jax import lax
from jax.random import uniform
from typing import Any, Callable, Sequence, Tuple
from functools import partial
import jax

Array = Any


def siren_init(weight_std, dtype):
    def init_fun(key, shape, dtype=dtype):
        if dtype == jnp.dtype(jnp.array([1j])):
            key1, key2 = jax.random.split(key)
            dtype = jnp.dtype(jnp.array([1j]).real)
            a = uniform(key1, shape, dtype) * 2 * weight_std - weight_std
            b = uniform(key2, shape, dtype) * 2 * weight_std - weight_std
            return a + 1j*b
        else:
            return uniform(key, shape, dtype) * 2 * weight_std - weight_std

    return init_fun

def siren_init2(min, max, dtype=jnp.float32):
    def init_fun(key, shape, dtype=dtype):
        if dtype == jnp.dtype(jnp.array([1j])):
            key1, key2 = jax.random.split(key)
            dtype = jnp.dtype(jnp.array([1j]).real)
            a = uniform(key1, shape, dtype, minval=min, maxval=max)
            b = uniform(key2, shape, dtype, minval=min, maxval=max)
            return a + 1j*b
        else:
            return uniform(key, shape, dtype, minval=min, maxval=max)
    return init_fun

def grid_init(grid_dimension, dtype):
    def init_fun(dtype=dtype):
        coord_axis = [jnp.linspace(-1, 1, d) for d in grid_dimension]
        grid = jnp.stack(jnp.meshgrid(*coord_axis), -1)
        return jnp.asarray(grid, dtype)

    return init_fun


class Sine(nn.Module):
    w0: float = 1.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        return jnp.sin(self.w0 * inputs)


class SirenLayer(nn.Module):
    features: int = 32
    w0: float = 1.0
    c: float = 6.0
    is_first: bool = False
    use_bias: bool = True
    act: Callable = jnp.sin
    precision: Any = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        input_dim = inputs.shape[-1]

        # Linear projection with init proposed in SIREN paper
        weight_std = (
            (1 / input_dim) if self.is_first else jnp.sqrt(self.c / input_dim) / self.w0
        )
        weight_min = (-1 / self.features) if self.is_first else (-jnp.sqrt(6 / self.features) / self.w0)
        weight_max = (1 / self.features) if self.is_first else (jnp.sqrt(6 / self.features) / self.w0)
        kernel = self.param(
            # "kernel", siren_init(weight_std, self.dtype), (input_dim, self.features)
            "kernel", siren_init2(weight_min, weight_max), (input_dim, self.features)
        )
        kernel = jnp.asarray(kernel, self.dtype)

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias:
            # bias = self.param("bias", uniform, (self.features,))
            bias = self.param("bias", zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        if self.act=='linear':
            return y
        else:
            return self.act(self.w0 * y)

class Siren(nn.Module):
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 4
    w0: float = 30.0
    w0_first_layer: float = 30.0
    use_bias: bool = True
    final_activation: Callable = lambda x: x  # Identity
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = jnp.asarray(inputs, self.dtype)
        
        for layernum in range(self.num_layers - 1):
            is_first = layernum == 0

            x = SirenLayer(
                features=self.hidden_dim,
                w0=self.w0_first_layer if is_first else self.w0,
                is_first=is_first,
                use_bias=self.use_bias,
            )(x)

        # Last layer, with different activation function
        x = SirenLayer(
            features=self.output_dim,
            w0=self.w0,
            is_first=False,
            use_bias=self.use_bias,
            act=self.final_activation,
        )(x)

        return x.squeeze()