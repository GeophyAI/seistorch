import jax.numpy as jnp
import jax
from jax import lax

# @jax.jit
def dynamic_set_zero(x, key, split=0.2):
    rng_key, subkey = jax.random.split(key)
    shift = jnp.arange(0, split * x.shape[0], dtype=jnp.int32)
    tau_s = jax.random.choice(key, shift)

    mask = jnp.arange(x.shape[0]) < tau_s
    mask = mask.reshape(-1, 1)
    x = x * (~mask)

    return tau_s, x, rng_key

rng_key = jax.random.PRNGKey(20240908)
for i in range(10):
    x = jnp.ones((1000,100))
    tau_s, x, rng_key = dynamic_set_zero(x, rng_key)

    print(tau_s)