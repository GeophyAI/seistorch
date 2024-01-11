# Seistorch: Where wave equations meets Automatic Differentiation

Seistorch is a seismic inversion framework designed for researchers in the field of geophysics and seismic imaging. This open-source Python library empowers users to perform state-of-the-art Full-Waveform Inversion (FWI) based on pytorch's automatic differentiation.

- **Key features** :

    1. **Python-Powered**: Seistorch is developed entirely in Python, ensuring ease of use and compatibility with various Python environments.

    2. **PyTorch backend**: The package leverages the automatic differentiation capabilities of PyTorch to efficiently compute gradients, a fundamental aspect of FWI.

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

# Implementation in Seistorch

We have integrated the work of many other scholars into Seistorch.

Such as the misfits in Seistorch: 
1. `seistorch.loss.Envelope`: [Chi B.X. et al., Envelope loss](https://linkinghub.elsevier.com/retrieve/pii/S0926985114002031). 

2. `seistorch.loss.Wasserstein1d`: [Yang & Engquist, Wasserstein loss](https://library.seg.org/doi/10.1190/geo2017-0264.1).

3. `seistorch.loss.NormalizedIntegrationMethod`: [Donno et al., Normalized Integration Method](http://www.earthdoc.org/publication/publicationdetails/?publication=69286)

4. `seistorch.loss.Traveltime`: [Luo & Schuster, Travel time misfit](https://library.seg.org/doi/10.1190/1.1443081).

5. Impicit neural network by [Sun J. et al.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JB025964).

6. Source encoding FWI by [Krebs et al.](https://doi.org/10.1190/1.3230502).

7. Gradient-descent based source inversion by [Felipe et al.](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/16th_CISBGf/session/FULL%20WAVEFORM%20INVERSION%20-%20FWI/Source%20wavelet%20estimation%20in%20FWI%20context.pdf).

8. Physical informed neural network by [Rasht-Behesht et al.](https://onlinelibrary.wiley.com/doi/10.1029/2021JB023120).

9. Boundary saving strategy by [Dussaud et al.](https://library.seg.org/doi/10.1190/1.3059336) and [Wang et al.](https://ieeexplore.ieee.org/document/10256076).

10. Integrating deep neural networks with full-waveform inversion by [Zhu et al.](https://doi.org/10.1190%2FGEO2020-0933.1).

11. AD-based acoustic inversion [Sun et al.](https://doi.org/10.1190/geo2019-0138.1).

12. AD-based elastic inversion [Wang W.L. et al.](https://doi.org/10.1190/geo2020-0542.1).

etc.

# TODO Recently
1. Random boundary by [Shen & Clapp](https://doi/10.1190/geo2014-0542.1).

2. A zoo of PINNs.

3. Gradient sampling.

4. FWIGAN.

5. Torched-based finite element method.

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
