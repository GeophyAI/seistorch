from jax.sharding import Mesh, PartitionSpec, NamedSharding, SingleDeviceSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax

devices = jax.devices()
devices_count = len(devices)

# Create a mesh and annotate each axis with a name.
device_mesh = mesh_utils.create_device_mesh((devices_count))
print(device_mesh)

mesh = Mesh(devices=device_mesh, axis_names=('data', ))
print(mesh)

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)

x_sharding = mesh_sharding(PartitionSpec('data', None)) # dimensions: (batch, length)
x = jnp.ones((2, 10))
print(x.devices(), x.sharding)
x = jax.device_put(x, x_sharding)
print(x.devices(), x.sharding)
for idx, devices in enumerate(jax.devices()):
  print(f"Device {idx}: {x.addressable_data(idx).shape}")
