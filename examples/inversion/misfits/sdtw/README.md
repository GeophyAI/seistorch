# Local coherence misfit
This example will reimplement the local coherence misfit proposed in paper [Full Waveform Inversion Using a High-Dimensional Local-Coherence Misfit Function, Yu et al(2023)](http://dx.doi.org/10.1109/TGRS.2023.3263501). It is a combination of the normalized cross-correlation and the structural similarity index(SSIM), which meaures the coherence between the windowed observed and synthetic data. 
# Theory
The local-coherence misfit function is defined as:
$$
J(\mathbf{m})=1-\frac{1}{N} \sum_{i=1}^{N} C\left(\mathbf{x}_r, t\right)
$$
where $N$ is the number of windows, $\mathbf{m}$ is the model parameter, $\mathbf{x}_r$ is the receiver position, and $t$ is the time. The local coherence $C$ is defined as:
$$
C\left(\mathbf{x}_r, t\right)=\frac{\tilde{D}_{\mathrm{syn}} \cdot \tilde{D}_{\mathrm{obs}} }{{|| \tilde{D}_{\mathrm{syn}} ||^2} {|| \tilde{D}_{\mathrm{obs}} ||^2}}
$$
where $\tilde{D}_{\mathrm{syn}}=D_{\mathrm{syn}}-\mathrm{mean}(D_{\mathrm{syn}})$ and $\tilde{D}_{\mathrm{obs}}=D_{\mathrm{obs}}-\mathrm{mean}(D_{\mathrm{obs}})$ are the moved data by subtracting their mean value, $D_{\mathrm{syn}}$ and $D_{\mathrm{obs}}$ are the synthetic and observed data after windowing and transformation, respectively. Since $\tilde D$ is 2D data, the dot product is calculated by flattening the 2D data to 1D data.

$\tilde D$ is calculated by convolving the windowed data with a 2D Gaussian window function $g$:
$$
\tilde D_{\text{syn}} = d^{\text{window}}_{\text{syn}} *g
$$
$$
\tilde D_{\text{obs}} = d^{\text{window}}_{\text{obs}} *g
$$
where $d^{\text{window}}_{\text{syn}}$ and $d^{\text{window}}_{\text{syn}}$ are the windowed synthetic and observed data, respectively, and $g$ is the Gaussian window function.

The Gaussian window function is defined as:
$$
g(h,\tau)=\frac{\Delta \tau \Delta h}{\sigma_t \sigma_h} \exp \left(-\frac{h^2}{2 \sigma_h^2}\right) \exp \left(-\frac{{\tau}^2}{2 \sigma_{\tau}^2}\right)
$$
where $\sigma$ is the standard deviation of the Gaussian window function, $\Delta h$ and $\Delta \tau$ are the spatial and temporal sampling interval, respectively.

# Example
For calculate the local coherence, we first simulate the observed data and initial data with ground-truth model and initial model, respectively. Then we calculate the local coherence using the script `show_lc.py`. 

![Ricker](figures/LocalCoherence_Shots.png)

In this script, a ricker example is also implemented to show the local coherence. The results are shown in the following figures.

![Ricker](figures/LocalCoherence_Ricker.png)

