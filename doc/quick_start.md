# Quick Start
Welcome to Seistorch! This quick start guide will walk you through the basics of using Seistorch for seismic simulations and Full-Waveform Inversion (FWI). 

**Note**: The FWI results demonstrated in our example are approximate and indicate incomplete inversion. To achieve better inversion results, you may need to adjust various parameters, such as the learning rate, decay rates, source wavelet frequencies, and potentially other hyperparameters. Fine-tuning these parameters can significantly impact the quality and convergence of the inversion process, allowing you to obtain more accurate subsurface models. It often requires experimentation and tuning to find the optimal set of parameters for your specific seismic data and geological conditions.

We'll cover the following topics:

1. [**Forward modeling**](forward_modeling.md): Simulate seismic wave propagation in 2D or 3D space.

2. [**Full waveform inversion**](fwi.md): This topic covers folowing fwi examples:
    - source-encoding-based fwi in [acoustic](fwi.md#2d-source-encoding-acoustic-fwi) and [elastic](fwi.md#2d-source-encoding-elastic-fwi) cases
    - [batched and mpi-based tradtional fwi](fwi.md#2d-batched-classic-fwi)
    - [memory-saving fwi using boundary-saving](fwi.md#boundary-saving-based-automatic-differentiation)
    - [towed acquistion fwi with marmousi model](fwi.md#towed-streamer-data-generation-and-inversion)

3. [**Reverse time migration**](rtm.md): This topic intorduce how to perform reverse time migration using seistorch.

4. [**Other usage**](other_examples.md): This topic covers following examples:

    - how to perform [source inversion](other_examples.md#source-inversion)
    - how to [calculate the adjoint source using torch](other_examples.md#how-to-calculate-the-adjoint-source-in-torch)
