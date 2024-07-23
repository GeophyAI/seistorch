# Seistorch: Where wave equations meets Automatic Differentiation

From this version, seistorch will use 'torchrun' to perform distributed full waveform inversion('mpi4py' used before). Please refer to <seistorch/examples/check_features/torchrun_dist>. The old mpi4py APIs in seistorch will be deprecated, nccl/mpi in torch will be prefered.

| Inversion Tests | Status |
| :----------- | :-----------: |
| Acoustic   | Passed       |
| Acoustic+NN   | Passed     |
| Elastic   | Passed   |
| Others   | Not test yet   |

# If you are interested in ...
## FWI with automatic differentiation
But you are a beginner of both FWI and pytorch, please refer to the [examples/easy_fwi](examples/sthelse/) folder. It provides a straightforward implementation of the simplest acoustic wave equation, allowing users to grasp the fundamental concepts of FWI.
## Physical Informed Neural Networks (PINN)
You wants to know something about how to train a neural network with physical constraints, please refer to the [examples/pinn](examples/pinn/) folder.
## Neural Representation of Velocity Model(Implicit FWI)
You want to know how to use neural network to represent the velocity model, please refer to the [examples/implicit_nn](examples/check_features/implicitNN_easy/) folder.
## Simulation with Seistorch
Please refer to the [examples/forward_modeling](examples/forward_modeling/different_eq/) folder.
## Perform FWI with Seistorch
Please refer to the [examples/source_encoding](examples/inversion/source_encoding/) folder or [examples/classic_fwi](examples/inversion/towed/towed_2d/).
## Perform LSRTM with Seistorch
Please refer to the [examples/lsrtm](examples/lsrtm/) folder.
## Popular loss functions in FWI
Please refer to the [examples/loss_functions](examples/inversion/misfits/) folder or refer to the source code in seistorch [seistorch/loss.py](seistorch/loss.py).
## Optimizers in FWI
You want to know how to use different optimizers for fwi with seistorch, please refer to the [examples/optimizers](examples/inversion/optimizers/) folder.

# New features:
| Type | New | Old |
| :----------- | ----------- | :-----------: |
| Boundary conditions   | HABC([Xie et al.](https://doi.org/10.1093/jge/gxz102))    | PML |
|Distributed FWI| [torchrun](https://pytorch.org/docs/stable/elastic/run.html) | [mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) |
|Anisotropic FWI| None |
|LSRTM|Elastic([Feng & Schuster])(10.1190/geo2016-0254.1)|None|
|LSRTM|Acoustic([Dai et al.])(10.1190/1.3513494)|None|

# Supported equations

| EQUATIONS | USAGE | REFERENCES|
| :-------------- | :-----------: | :------------------|
| Acoustic (2nd) | FWI | * |
| Acoustic (2nd) | LSRTM | 10.1190/1.3513494 |
| Acoustic (1st) | FWI | * |
| qP VTI (2nd) | FWI | 10.1190/geo2022-0292.1 |
| qP TTI (2nd) | FWI | 10.1190/geo2022-0292.1 |
| ViscoAcoustic  (2nd) | FWI | 10.3997/2214-4609.201601578 |
| ViscoAcoustic2  (2nd) | FWI | 10.3997/2214-4609.201402310 |
| Elastic (1st)   | FWI | 10.1190/1.1442147 |
| Elastic (1st)   | LSRTM | 10.1190/geo2016-0254.1 |
| TTI-Elastic (1st)  | FWI | * |
| Acoustic-Elastic coupled (1st) | FWI | 10.1190/geo2015-0535.1 |
| Velocity-Dilatation-Rotation (1st) | FWI | 10.1190/geo2016-0245.1 | 

Note: 2nd means displacement equations, 1st means velocity-stress equations.

# Citation

If you find this work useful for your research, please consider citing our paper [Memory Optimization in RNN-based Full Waveform Inversion using Boundary Saving Wavefield Reconstruction](https://ieeexplore.ieee.org/document/10256076):

```
@ARTICLE{10256076,
  author={Wang, Shaowen and Jiang, Yong and Song, Peng and Tan, Jun and Liu, Zhaolun and He, Bingshou},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Memory Optimization in RNN-based Full Waveform Inversion using Boundary Saving Wavefield Reconstruction}, 
  year={2023},
  volume={61},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3317529}}
```
