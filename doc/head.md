# Introduction

**Seistorch: Python Package for Full-Waveform Inversion (FWI)**

Seistorch is a powerful Python package designed for Full-Waveform Inversion (FWI) in the field of geophysics. FWI is a crucial technique used to recover subsurface properties by comparing observed and simulated seismic waveforms. Seistorch simplifies and streamlines the FWI process, making it accessible to researchers and professionals in the Earth Sciences.

## Key Features

- **Python-Powered**: Seistorch is developed entirely in Python, ensuring ease of use and compatibility with various Python environments.

- **Automatic Differentiation with PyTorch**: The package leverages the automatic differentiation capabilities of PyTorch to efficiently compute gradients, a fundamental aspect of FWI.

- **Parallel Computing using multiple nodes and GPUs**: Seistorch is equipped with a powerful feature that allows for parallel computing using multiple nodes and multiple GPUs. This capability can significantly accelerate your Full-Waveform Inversion (FWI) and forward modeling tasks. To utilize this feature effectively, please follow the guidelines in [**Getting Started**](configure.md).

- **Computational Load Balancing**: Seistorch's MPI load balancing optimizes task allocation across nodes, maximizing computational resource utilization and speeding up simulations.

- **Memory saving strategy**: Seistorch offers a powerful feature that allows users to control and reduce GPU memory consumption through *boundary saving strategies*. This feature is particularly useful when working with large-scale seismic simulations.

- **Multiple FWI Variants**: Seistorch supports both MPI-based Classic FWI (The gradients are accumulated from each shot) and Source Encoding FWI (the gradient is calculated by a supershot), giving users the flexibility to choose the approach that suits their needs.

- **User-Friendly**: The package comes with an intuitive API and extensive documentation, making it accessible to both beginners and experienced geophysicists.

## Dependencies

1. **Python Packages**: Before you can start using Seistorch, you need to ensure that you have the following Python packages installed:
    - numpy
    - torch
    - mpi4py
    - pot
    - tensorboard
    - torchvision
    - tqdm
    - pyyaml

2. **MPICH 4.1.1**: Seistorch relies on MPICH (Message Passing Interface) for parallel computing when performing Classic FWI. Please ensure you have MPICH version 4.1.1 installed on your system. Other versions, for instance 1.4.1, are not compatible with Seistorch.


## Acknowledgements

We would like to express our sincere gratitude to the contributors and developers of the open-source project [wavetorch](https://github.com/fancompute/wavetorch). Their work has been a significant source of inspiration and has contributed to the development of Seistorch.

The wavetorch project has provided valuable insights and innovative solutions in the field of computational seismology and Full-Waveform Inversion (FWI). Many aspects of Seistorch, including its functionality and certain function naming conventions (such as `WaveCell`, `WaveRNN`, `WaveSource`, `WaveProbe`, etc), have been inspired by the ideas and codebase of wavetorch.

## Reference

T. W. Hughes*, I. A. D. Williamson*, M. Minkov, and S. Fan, [Wave physics as an analog recurrent neural network](https://advances.sciencemag.org/content/5/12/eaay6946), Science Advances, vol. 5, no. 12, p. eaay6946, Dec. 2019

Sun, J., Innanen, K., Zhang, T., & Trad, D. (2023). [Implicit seismic full waveform inversion with deep neural representation](https://doi.org/10.1029/2022JB025964). Journal of Geophysical Research: Solid Earth, 128, e2022JB025964.


## Citation

If you find this work useful for your research, please consider citing our paper [Memory Optimization in RNN-based Full Waveform Inversion using Boundary Saving Wavefield Reconstruction](https://ieeexplore.ieee.org/document/10256076):

```
@ARTICLE{10256076,
  author={Wang, Shaowen and Jiang, Yong and Song, Peng and Tan, Jun and Liu, Zhaolun and He, Bingshou},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Memory Optimization in RNN-based Full Waveform Inversion using Boundary Saving Wavefield Reconstruction}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3317529}}
```