from jax.experimental import io_callback
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from jax.experimental import host_callback as hcb

# global_rng = np.random.default_rng(0)

# def host_side_random_like(x, step):
#   """Generate a random array like x using the global_rng state"""
#   # We have two side-effects here:
#   # - printing the shape and dtype
#   # - calling global_rng, thus updating its state
#   print(f'generating {x.dtype}{list(x.shape)}')
#   jnp.save(f'tmp/x{step}.npy', x)
#   return x

# @jax.jit
# def numpy_random_like(x):
#     io_callback(host_side_random_like, x, x, 0)
#     return 1

# x = jnp.zeros(5)
# numpy_random_like(x)

def f(x):
    return x**2+1

def fwd(x, cstep):
    y = f(x)
    io_callback(save2file, x, x, cstep)
    return y, None

def bwd(res, g):
    x, cstep = res
    io_callback(loadfromfile, x, x, cstep)
    return g, None

f = jax.custom_vjp(f)
f.defvjp(fwd, bwd)

def save2file(data, cstep):
    jnp.save(f'tmp/x{cstep}.npy', data)
    return data

def loadfromfile(cstep):
    return jnp.load(f'tmp/x{cstep}.npy')

def step(x, cstep):
    # _f.defvjp(fwd, bwd)
    y = f(x)
    return y, None

x = jnp.array([1., 2.])
nt = 5

# loop over using scan
initial_carry = x
csteps = jnp.arange(10)
final_carry, _ = lax.scan(step, initial_carry, jnp.arange(nt))

def loss(x):
    return jnp.sum(x)
# compute grad
@jax.jit
def cal_grad(x):
    return jax.value_and_grad(loss)(x)

loss, grad = cal_grad(final_carry)

print(grad)
