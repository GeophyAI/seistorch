# Instantaneous Phase Misfit
This misfit quantifies the difference between the instantaneous phase of the observed and synthetic data. The misfit is defined as:

$$
J(m) = \frac{1}{2} \sum_{s,r} \left( \phi^{obs} - \phi^{syn} \right)^2
$$
    
where $\phi^{obs}$ and $\phi^{syn}$ are the instantaneous phase of the observed and synthetic data, respectively.

The instantaneous phase is computed using the Hilbert transform. 
$$
\phi(t) = \arctan \left( \frac{H(d(t))}{d(t)} \right)
$$
where $d(t)$ is the data and $H(d(t))$ is the Hilbert transform of the data. For checking the hilbert transform codes of seistorch, please refer to the [Hilbert Transform](../../tests/test_hilber.py).

# Usage
```bash
# 1. Generate the Geometry and models
python generate_model_geometry.py
# 2. Generate the observed data
sh forward_with_true_vel.sh
# 3. Generate the synthetic data
sh forward_with_init_vel.sh
# 4. Compute the misfit and adjoint source
python cal_adj.py

```