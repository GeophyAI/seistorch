# Neural Network for Model Representation
This example demonstrates how to use a neural network to represent the model parameters in FWI. This method can also be called as [implicit FWI(Sun et al., 2023)](https://doi.org/10.1029/2022JB025964). We first use neural networks to represent the model parameters of seismic, then input them to the wave equation for simulating the wave propagation. The neural network is optimized by minimizing the misfit between the observed and predicted data. 

So, by model representation, we convert the seismic inversion problem to a neural network optimization problem, and it's also physics-guided, because the solver still be the wave equation.
# Theory
Wave equations are mainly solved by finite-difference(FDM) or finite-element(FEM) methods. These methods require a discretized model, i.e. grid-based model. For example, in FDM, the grid-based model $c$ in 2D case can be represented as
$$
\mathbf c = 
\begin{bmatrix}
c_{11} & c_{12} & \cdots & c_{1n} \\
c_{21} & c_{22} & \cdots & c_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
c_{m1} & c_{m2} & \cdots & c_{mn}
\end{bmatrix}
$$
where $c_{ij}$ is the model parameter at the $i$-th row and $j$-th column. The wave equation is then solved on this grid-based model.

Actually, we can use a neural network to represent these model parameters, which is called the model representation. For example, we can map coordinates $(x,y,z)$ to the model parameters $(c_{11},c_{12},c_{13})$ by a neural network. The model representation can be written as
$$
\mathbf c = F(x,y,z; \theta)
$$
where $F$ is the neural network, $(x,y,z)$ are the spatial coordinates, and $\theta$ are the parameters of the neural network.
