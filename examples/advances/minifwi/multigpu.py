from jax import random, pmap
import jax.numpy as jnp

# Jax on multiple gpus

# Create 8 random 5000 x 6000 matrices, one per GPU
keys = random.split(random.PRNGKey(0), 2)
mats = pmap(lambda key: random.normal(key, (5000, 6000)))(keys)
print(mats.devices())
# Run a local matmul on each device in parallel (no data transfer)
result = pmap(lambda x: jnp.dot(x, x.T))(mats)  # result.shape is (8, 5000, 5000)

# Compute the mean on each device in parallel and print the result
print(pmap(jnp.mean)(result))
# prints [1.1566595 1.1805978 ... 1.2321935 1.2015157]