import jax
import jax.numpy as jnp
import numpy as np
import torch
import timeit

def inplace_fn_jax(x):
    for _ in range(100):
        x = x + x * x + x * x * x
        x = x.at[0, 0].set(0)
    return x

def inplace_fn_torch(x):
    for _ in range(100):
        x = x + x * x + x * x * x
        x[0, 0] = 0
    return x
    
y = np.random.randn(1000, 1000).astype(dtype='float32')
y_torch = torch.tensor(y)#.cuda()
y_jax = jnp.array(y)

jax_fn = jax.jit(inplace_fn_jax)
jax_fn(y_jax).block_until_ready()

print("With inplace operations:")

tt = timeit.timeit(lambda: inplace_fn_torch(y_torch), number=10)
print(f"PyTorch: {tt * 1000} msec")

t = timeit.timeit(lambda: jax_fn(y_jax).block_until_ready(), number=10)
print(f"JAX: {t * 1000} msec")


def noinplace_fn_jax(x):
    for _ in range(100):
        x = x + x * x + x * x * x
    return x


jax_fn = jax.jit(noinplace_fn_jax)
jax_fn(y_jax).block_until_ready()

print("Without inplace operations:")

tt = timeit.timeit(lambda: noinplace_fn_jax(y_torch), number=10)
print(f"PyTorch: {tt * 1000} msec")

t = timeit.timeit(lambda: jax_fn(y_jax).block_until_ready(), number=10)
print(f"JAX: {t * 1000} msec")
