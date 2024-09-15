# Acoustic FWI (Source Encoding)
# Usage
This example provides both jax-based and torch-based implementations of the acoustic FWI with source encoding.

Please note that the jax-based implementation only support `.hdf5` format for the observed data, while the torch-based implementation only support `.npy` format, at least for now.

```bash
# Jax-based implementation
# 1. Generate model geometry
python generate_model_geometry.py
# 2. Generate observed data
sh forward_hdf5.sh
# 3. Run inversion
sh sefwi_jax.sh
```

```bash
# Torch-based implementation
# 1. Generate model geometry
python generate_model_geometry.py
# 2. Generate observed data
sh forward_npy.sh
# 3. Run inversion
sh sefwi_torch.sh
```