# Seistorch: Where wave equations meets Automatic Differentiation

From this version, seistorch will use 'torchrun' to perform distributed full waveform inversion('mpi4py' used before). Please refer to <seistorch/examples/check_features/torchrun_dist>. The old mpi4py APIs in seistorch will be deprecated, nccl in torch will be prefered.

Note: I am preparing my dissertation now, so the update may be delayed.

| Inversion Tests | Status |
| :----------- | :-----------: |
| Acoustic   | Passed       |
| Acoustic+NN   | Passed     |
| Elastic   | Not test yet   |
| Others   | Not test yet   |

# New features:
| Type | New | Old |
| :----------- | ----------- | :-----------: |
| Boundary conditions   | HABC([Xie et al.](https://doi.org/10.1093/jge/gxz102))    | PML |
|Distributed | [torchrun](https://pytorch.org/docs/stable/elastic/run.html) | [mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) |

# Citation

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
