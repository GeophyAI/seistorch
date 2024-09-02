# Seistorch: Where wave equations meets Automatic Differentiation

From this version, seistorch will use 'torchrun' to perform distributed full waveform inversion('mpi4py' used before). Please refer to <seistorch/examples/check_features/torchrun_dist>. The old mpi4py APIs in seistorch will be deprecated, nccl/mpi in torch will be prefered.

| Inversion Tests | Status |
| :----------- | :-----------: |
| Acoustic   | Passed       |
| Acoustic+NN   | Passed     |
| Elastic   | Passed   |
| Others   | Not test yet   |

# If you are interested in ...

I reproduced the results of the following papers using Seistorch and some stand alone codes. If you are interested in these topics, please refer to the following links:

### Forward modeling
| Traditional | Codes | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|Simulations|[click](examples/forward_modeling/different_eq)|[Wang et al., 2023](https://doi.org/10.1109/TGRS.2023.3317529)|Seistorch|
|Finite difference method|[click](examples/forward_modeling/pseudospectral)|-|Acoustic|
|Pseudospectral method|[click](examples/forward_modeling/pseudospectral)|[Kosloff & Baysal](https://doi.org/10.1190/1.1441288)|Acoustic|

### FWI
| Traditional | Codes | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|FWI by Pytorch|[click](examples/sthelse)|-|Stand alone|
|Acoustic FWI|[click](examples/inversion/source_encoding/acoustic)|-|Seistorch, Source Encoding|
|Elastic FWI|[click](examples/inversion/source_encoding/elastic)|-|Seistorch|
|Regularization-based FWI|[click](examples/regularization/model_reg_fwi)|||

### LSRTM
| Traditional | Codes | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|Acoustic LSRTM|[click](examples/lsrtm)|[Dai et al., 2010](https://doi.org/10.1190/1.3513494)|Seistorch|
|Elastic LSRTM|[click](examples/lsrtm)|[Feng & Schuster, 2017](https://doi.org/10.1190/geo2016-0254.1)|Seistorch|
|VTI/TTI LSRTM|[click](examples/lsrtm/qP)|-|Seistorch|
|Joint FWI&LSRTM|[click](examples/inversion/joint_fwi_lsrtm)|[Wu et al., 2024](https://doi.org/10.1109/TGRS.2024.3349608)|Seistorch|
|Regularization-based LSRTM|[click](examples/regularization/model_reg_lsrtm)|||
### Inversion with Neural Networks
| FWI+NeuralNetworks | Codes | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|PINN|[click](examples/pinn)|[Majid et al., 2022](https://doi.org/10.1029/2021JB023120)|Stand alone|
|Implicit FWI|[click](examples/nn_embedded_fwi/model_representation/implicit_acoustic)|[Sun et al., 2023](https://doi.org/10.1029/2022JB025964)|Stand alone <br> Model Reparameterization(Acoustic)|
|Physics-guided NN FWI|[click](examples/nn_embedded_fwi/model_representation/encoder_decoder_acoustic)|[Dhara & Sen, 2022](https://doi.org/10.1190/tle41060375.1)|Stand alone <br> Model Reparameterization(Acoustic)|
|Elastic parameters crosstalk|[click](examples/nn_embedded_fwi/model_representation/implicit_elastic)|-|Stand alone <br> Model Reparameterization(Elastic)|
|Siamese FWI|[click](examples/nn_embedded_fwi/siamesefwi)|[Omar et al., 2024](https://doi.org/10.1029/2024JH000227)|Stand alone|
|Elastic parameters crosstalk|[click](examples/nn_embedded_fwi/model_representation/encoder_decoder_elastic)|[Dhara & Sen](https://doi.org/10.1109/TGRS.2023.3294427)|Stand alone <br> Model Reparameterization(Elastic)|

### Misfit functions
| Misfits | Examples | Related Papers | Notes |
| :----------- | ----------- | :-----------: | :----- |
|Optimal Transport|[click](examples/inversion/misfits/ot)|[Yang & Ma, 2023](https://doi.org/10.1029/2022JB025493)<br>[Yang & Enguist](https://doi.org/10.1190/GEO2017-0264.1)|-|
|Envelope|[click](examples/inversion/misfits/envelope)|[Chi et al., 2014](https://doi.org/10.1016/j.jappgeo.2014.07.010) <br> [Wu et al., 2014](https://doi.org/10.1190/GEO2013-0294.1)|-|
|Traveltime|[click](examples/inversion/misfits/travel_time_misfit)|[Wang et al., 2024](https://doi.org/10.3997/2214-4609.202410170)|Differentiable|
|Cosine Similarity|[click](examples/inversion/misfits/cs)|[Choi & Alkhalifah, 2012](https://doi.org/10.1111/j.1365-2478.2012.01079.x)<br>[Liu et al., 2016](https://doi.org/10.1093/gji/ggw485)<br>|Global correlation<br>Normalized zero-lag cross-correlation|
|L1|[click](examples/inversion/misfits/l1)|||
|Local coherence|[click](examples/inversion/misfits/localcoherence)|[Yu et al., 2023](https://doi.org/10.1109/TGRS.2023.3263501)||
|Instantaneous Phase|[click](examples/inversion/misfits/ip)|[Bozdag et al., 2011](https://doi.org/10.1111/j.1365246X.2011.04970.x) <br> [Yuan et al., 2020](https://doi.org/10.1093/gji/ggaa063)||
|Weighted loss|[click](examples/inversion/misfits/weighted)|[Song et al., 2023](https://doi.org/10.1109/TGRS.2023.3300127)||
|Envelope Cosine Similarity|*|[Oh and Alkhalifah, 2018](https://doi.org/10.1093/gji/ggy031)|Envelope-based Global Correlation Norm|
|Soft Dynamic Time warpping|[click](examples/inversion/misfits/sdtw)|[Maghoumi, 2020](https://stars.library.ucf.edu/etd2020/379/)<br>[Maghoumi et al., 2020](https://arxiv.org/abs/2011.09149)||

# New features:
| Type | New | Old |
| :----------- | ----------- | :-----------: |
|Weighted loss|Inspired by [Song et al., 2023](https://doi.org/10.1109/TGRS.2023.3300127)||
| Boundary conditions   | HABC([Xie et al.](https://doi.org/10.1093/jge/gxz102))    | PML |
|Distributed FWI| [torchrun](https://pytorch.org/docs/stable/elastic/run.html) | [mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) |
|Anisotropic FWI| None |
|LSRTM|Elastic [Feng & Schuster](10.1190/geo2016-0254.1)|None|
|LSRTM|Acoustic [Dai et al.](10.1190/1.3513494)|None|

# Supported equations

| EQUATIONS | USAGE | REFERENCES| EQUATION CODES |
| :-------------- | :-----------: | :------------------| :-----------: |
| Scalar Acoustic (2nd) | FWI | * | [PML version](seistorch/equations2d/acoustic.py) <br> [HABC version](seistorch/equations2d/acoustic_habc.py) |
| Scalar Acoustic (2nd) | LSRTM | [Dai et al., 2010](https://doi.org/10.1190/1.3513494) | [click](seistorch/equations2d/acoustic_habc.py) |
| Acoustic (1st) | FWI | * | [click](seistorch/equations2d/acoustic1st.py) |
|Variable Density (2nd)| FWI | [Whitmore et al., 2020](https://doi.org/10.3997/2214-4609.202010332) | [click](seistorch/equations2d/acoustic_rho_habc.py) |
| Joint FWI & LSRTM| FWI+LSRTM | [Wu et al., 2024](https://doi.org/10.1109/TGRS.2024.3349608) | [click](seistorch/equations2d/acoustic_fwim_habc.py) |
| qP TTI (2nd) | FWI/LSRTM | [Liang et al., 2024](https://doi.org/10.1190/geo2022-0292.1) | [fwi click](seistorch/equations2d/tti_habc.py) <br> [lsrtm click](seistorch/equations2d/acoustic_tti_lsrtm_habc.py) |
| qP VTI (2nd) | FWI/LSRTM | [Liang et al., 2024](https://doi.org/10.1190/geo2022-0292.1) | [fwi click](seistorch/equations2d/vti_habc2.py) <br> [lsrtm click](seistorch/equations2d/acoustic_vti_lsrtm_habc.py)  |
| ViscoAcoustic  (2nd) | FWI | [Li et al., 2016](https://doi.org/10.3997/2214-4609.201601578) | [click](seistorch/equations2d/vacoustic_habc.py) |
| VTI  (2nd) | FWI | [Zhou et al., 2006](https://doi.org/10.3997/2214-4609.201402310) | [click](seistorch/equations2d/vti_habc.py) |
| Elastic (1st)   | FWI | [Virieux, 1986](https://doi.org/10.1190/1.1442147) | [click](seistorch/equations2d/elastic.py) |
| Elastic (1st)   | LSRTM | [Feng & Schuster, 2017](https://doi.org/10.1190/geo2016-0254.1) | [click](seistorch/equations2d/elastic_lsrtm.py) |
| TTI-Elastic (1st)  | FWI | * | [click](seistorch/equations2d/ttielastic.py) |
| Acoustic-Elastic coupled (1st) | FWI | [Yu et al., 2016](https://doi.org/10.1190/geo2015-0535.1) | [click](seistorch/equations2d/aec.py) |
| Velocity-Dilatation-Rotation (1st) | FWI | [Tang et al., 2016](https://doi.org/10.1190/geo2016-0245.1) | [click](seistorch/equations2d/vdr.py) |

Note: 2nd means displacement equations, 1st means velocity-stress equations.

# To do list
- [SH- and Love-wave FWI(Dokter et al., 2017)](https://doi.org/10.1111/1365-2478.12549)

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
