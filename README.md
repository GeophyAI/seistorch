# Seistorch: Where wave equations meets Automatic Differentiation

From this version, seistorch will use 'torchrun' to perform distributed full waveform inversion('mpi4py' used before). Please refer to <seistorch/examples/check_features/torchrun_dist>. The old mpi4py APIs in seistorch will be deprecated, nccl/mpi in torch will be prefered.

| Inversion Tests | Status |
| :----------- | :-----------: |
| Acoustic   | Passed       |
| Acoustic+NN   | Passed     |
| Elastic   | Passed   |
| Others   | Not test yet   |

# If you are interested in ...
### Traditional methods
| Traditional | Codes | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|FWI by Pytorch|[click](examples/sthelse)|-|Stand alone|
|Simulations|[click](examples/forward_modeling/different_eq)|[10.1109/TGRS.2023.3317529](https://doi.org/10.1109/TGRS.2023.3317529)|Seistorch|
|Acoustic LSRTM|[click](examples/lsrtm)|[10.1190/1.3513494](https://doi.org/10.1190/1.3513494)|Seistorch|
|Elastic LSRTM|[click](examples/lsrtm)|[10.1190/geo2016-0254.1](https://doi.org/10.1190/geo2016-0254.1)|Seistorch|
|Acoustic FWI|[click](examples/inversion/source_encoding/acoustic)|-|Seistorch, Source Encoding|
|Elastic FWI|[click](examples/inversion/source_encoding/elastic)|-|Seistorch|
|Joint FWI&LSRTM|[click](examples/inversion/joint_fwi_lsrtm)|[10.1109/TGRS.2024.3349608](https://doi.org/10.1109/TGRS.2024.3349608)|Seistorch|
### Inversion with Neural Networks
| FWI+NeuralNetworks | Codes | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|PINN|[click](examples/pinn)|[10.1029/2021JB023120](https://doi.org/10.1029/2021JB023120)|Stand alone|
|Model Reparameterization|[click](examples/check_features/implicitNN)|[10.1029/2022JB025964](https://doi.org/10.1029/2022JB025964)|Stand alone|
|Siamese FWI|[click](examples/nn_embedded_fwi/siamesefwi)|[10.1029/2024JH000227](https://doi.org/10.1029/2024JH000227)|Stand alone|
### Misfit functions
| Misfits | Examples | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|Optimal Transport|[click](examples/inversion/misfits/ot)|[10.1029/2022JB025493](https://doi.org/10.1029/2022JB025493)<br>[10.1190/GEO2017-0264.1](https://doi.org/10.1190/GEO2017-0264.1)|-|
|Envelope|[click](examples/inversion/misfits/envelope)|[10.1016/j.jappgeo.2014.07.010](https://doi.org/10.1016/j.jappgeo.2014.07.010)|-|
|Traveltime|[click](examples/inversion/misfits/travel_time_misfit)|[10.3997/2214-4609.202410170](https://doi.org/10.3997/2214-4609.202410170)|Differentiable|
|Cosine Similarity|[click](examples/inversion/misfits/cs)|[10.1111/j.1365-2478.2012.01079.x](https://doi.org/10.1111/j.1365-2478.2012.01079.x)<br>[10.1093/gji/ggw485](https://doi.org/10.1093/gji/ggw485)<br>|Global correlation<br>Normalized zero-lag cross-correlation|
|L1|[click](examples/inversion/misfits/l1)|||
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
| Scalar Acoustic (2nd) | FWI | * |
| Scalar Acoustic (2nd) | LSRTM | 10.1190/1.3513494 |
| Acoustic (1st) | FWI | * |
|Variable Density (2nd)| FWI | * |
| Joint FWI & LSRTM|FWI+LSRTM |10.1109/TGRS.2024.3349608|
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
