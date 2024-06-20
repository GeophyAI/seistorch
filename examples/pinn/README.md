# How to run this example

This is an example of showing how PINN (physical informed neural network) works. The PINN is a data-driven solutions of partial differential equations proposed by [Raissi et al., 2018](https://doi.org/10.1016/j.jcp.2018.10.045).

1. Run `generate_model_geometry.py` for generating the velocity model and geometry files.
2. In this step, the script `forward.py` will help you model snapshots of pressure wavefields. The data will be stored in a folder named `wavefields`. These wavefield are training data for PINN.
3. Run the cells in `pinn.ipynb` and enjoy your PINN trip.