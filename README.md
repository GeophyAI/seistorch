# Seistorch: Advanced Seismic Inversion Framework

Seistorch is a seismic inversion framework designed for researchers in the field of geophysics and seismic imaging. This open-source Python library empowers users to perform state-of-the-art Full-Waveform Inversion (FWI) based on pytorch's automatic differentiation.

- **Key features**:

    1. **Python-Powered**: Seistorch is developed entirely in Python, ensuring ease of use and compatibility with various Python environments.

    2. **Automatic Differentiation with PyTorch**: The package leverages the automatic differentiation capabilities of PyTorch to efficiently compute gradients, a fundamental aspect of FWI.

    3. **Parallel Computing using multiple nodes and GPUs**: Seistorch is equipped with a powerful feature that allows for parallel computing using multiple nodes and multiple GPUs. This capability can significantly accelerate your Full-Waveform Inversion (FWI) and forward modeling tasks. To utilize this feature effectively, please follow the guidelines in Getting Started.

    4. **Computational Load Balancing**: Seistorch’s MPI load balancing optimizes task allocation across nodes, maximizing computational resource utilization and speeding up simulations.

    5. **Memory saving strategy**: Seistorch offers a powerful feature that allows users to control and reduce GPU memory consumption through boundary saving strategies. This feature is particularly useful when working with large-scale seismic simulations.

    6. **Multiple FWI Variants**: Seistorch supports both MPI-based Classic FWI (The gradients are accumulated from each shot) and Source Encoding FWI (the gradient is calculated by a supershot), giving users the flexibility to choose the approach that suits their needs.

    7. **User-Friendly**: The package comes with an intuitive API and extensive documentation, making it accessible to both beginners and experienced geophysicists.

- [More about Seistorch](https://seistorch.readthedocs.io/en/latest/)
    - [Introduction](https://seistorch.readthedocs.io/en/latest/head.html)
    - [Quick start (Examples)](https://seistorch.readthedocs.io/en/latest/quick_start.html)
    - [Getting Started](https://seistorch.readthedocs.io/en/latest/getting_started.html)
    - [Configurations](https://seistorch.readthedocs.io/en/latest/configure.html)
    - [Data format of seistorch](https://seistorch.readthedocs.io/en/latest/data_format.html)
    - [Configuration Parameters for FWI Scripts](https://seistorch.readthedocs.io/en/latest/running_commands.html)
    - [Advanced supports](https://seistorch.readthedocs.io/en/latest/advanced.html)
