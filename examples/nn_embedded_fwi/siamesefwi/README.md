# Siamese Full waveform inversion
This example reproduce the siamese fwi ([Omar et al., 2024](https://doi.org/10.1029/2024JH000227)), which use a network to map the data into a latent space and then compare them. The network is trained to minimize the difference between the latent space of the predicted and observed data.

# Theory
For conventional FWI, the objective function is defined as the difference between the observed and synthetic data:
$$
J(\mathbf m) = \frac{1}{2} \sum_{s,r}^{} \left\| d_{\text{obs}}^i - d_{\text{syn}}(\mathbf m)^i \right\|^2
$$
The siamese fwi, on the other hand, uses a network to map the data into a latent space, so we will have:
$$
\mathbf z_{obs}^{latent} = \mathcal F(\mathbf d_\text{obs}; \theta)
$$
and 
$$
\mathbf z_{syn}^{latent} = \mathcal F(\mathbf d_\text{syn}; \theta)
$$
where $\mathcal F$ is the network and $\theta$ are the network parameters. Both synthetic and observed data are mapped into a same latent space with a single network, that is why it is called siamese. 

The objective function of siamese fwi is defined as the difference between the latent space of the predicted and observed data:
$$
J(\mathbf m) = \frac{1}{2} \sum_{s,r}^{} \left\| \mathbf z_{\text{obs}}^{latent} - \mathbf z_{\text{syn}}^{latent} \right\|^2
$$

# Run the example
The configuration file is `configure.py`. You can change the parameters in this file to test the example.

First, we need to simulate the observed. The script `forward.py` is used to that.
```bash
python forward.py
```
Scripts `fwi_classic.py` and `fwi_siamese.py` are used to run the conventional L2-based and Siamese-based FWI, respectively. You can either run it in an interactive window for checking the intermediate results or run it in the background.


# Note
I only have one 2080Ti, cannot test the full Marmousi model, so I use a downsampled for testing. This test still to be done.

