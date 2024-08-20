# Finite-difference v.s. Pseudospectral

This example demonstrates the difference between the finite-difference and pseudospectral methods for solving the acoustic wave equation.

# Scalar Acoustic Wave Equation
The scalar acoustic wave equation is given by

$$
\frac{1}{c^2} \frac{\partial^2 p}{\partial t^2} = \nabla^2 p + f,
$$

where $p$ is the pressure field, $c$ is the wave velocity, and $f$ is the source term.

For calculating the spatial derivatives, the finite-difference method uses a finite difference stencil, while the pseudospectral method uses the Fourier transform.

$$
\nabla^2 p = -\mathcal{F}^{-1} \left( (k_x^2+k_y^2) \mathcal{F}(p) \right),
$$
    
where $\mathcal{F}$ is the 2D spatial Fourier transform, and $k_x$ and $k_y$ are the wavenumbers in the $x$ and $y$ directions, respectively.

# Variable density acoustic wave equation
The variable density acoustic wave equation is given by

$$
\frac{1}{c^2\rho} \frac{\partial^2 p}{\partial t^2} = \nabla \cdot (\frac{1}{\rho}\nabla p) + f,
$$

where $\rho$ is the density field, and $\nabla \cdot$ is the divergence operator.

In frequency domain, the divergence operator is given by
$$
\nabla \cdot \mathbf{A} = i(k_x \hat{A}(k_x,k_y) + k_y \hat{A}(k_x,k_y)),
$$

where $\mathbf{A}$ is a vector field, and $\hat{A}$ is the Fourier transform of $\mathbf{A}$.

# Implementation
For acoustic simulations, we need 2D fourier transforms twice in each time step with the pseudospectral method. While the finite-difference method only requires 2D spatial convolution.

The implementations can be found in script `utils_torch.py`. The functions named `laplace_ps` and `laplace_fd` are the implementations of the Laplacian operator in the PseudoSpectral and Finite-Difference methods, respectively.
