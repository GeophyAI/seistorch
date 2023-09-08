# Quick Start
Welcome to Seistorch! This quick start guide will walk you through the basics of using Seistorch for seismic simulations and Full-Waveform Inversion (FWI). We'll cover the following topics:

1. **Running Forward Modeling**: Simulate seismic wave propagation.

2. **Classic FWI**: Perform Full-Waveform Inversion.

3. **Source encoding FWI**: Learn how to perform source-encoding based FWI.

## Perform forward modeling

The code of this section locates at `examples/forward_modeling`. This example shows how to run forward modeling with your own model and geometry.

-   First we need to change the directory into it.

    ```shell
    cd examples/forward_modeling
    ```

-   Generate a two layer model and the corresponding sources-receivers pairs by typing:

    ```shell
    python generate_model_geometry.py
    ```

    Two new folders **geometry** and **velocity_model** will be created. The figure **model_geometry.png** illustrates the generated layer model and the locations of source-receiver pairs.

-   Running the shell script `forward.sh`, a file called `shot_gather.npy` will be created.

    ```shell
    sh forward.sh
    ```

-   Show the shot gathers.

    ```shell
    python show_shotgather.py
    ```

    The plotted results will be save in **shot_gather.png**.

## Perform acoustic full waveform inversion

The code of this section locates at `examples/acoustic_fwi`. This exmaples shows a workflow of performing full waveform inversion based on pure automatic differentiation (PAD) and boundary saving-based automatic differentiation (BSAD). The BSAD method is used to reduce the GPU memory usage by reconstructing the wavefield with boundary saving strategy during loss backpropagation.

- **Generate model and geometry**

    ```shell
    python generate_model_geometry.py
    ```

    The srcipt `generate_model_geometry.py` will generate a 2 layer ground truth model and a smoothed version of it. The corresponding source-receiver pairs will be generated as well. A figure named **model_geometry.png** illustrate the true and initial model.

- **Generate observed data**

    The configure file of forward modeling lies in `configs/forward.yml`

    ```shell
    sh forward.sh
    ```

    The script `forward.sh` will genereate the observed data **shot_gather.npy** for inversion.

- **Perform PAD FWI and BSAD FWI**

    Set `boundary_saving: true` can reduce GPU memory usage.

    Perform full waveform inversion with pure automatic differeniation.

    ```shell
    sh inversion_AD.sh
    ```

    Perform full waveform inversion with boundary-saving automatic differeniation.

    ```shell
    sh inversion_BS.sh
    ```

    The configure files can be found in `acoustic_fwi/configs/inversion_AD.yml` and `acoustic_fwi/configs/inversion_BS.yml`.

    Compare the results between the two methods.

    ```shell
    python AD_vs_BS.py
    ```

    The BSAD method significantly reduces memory usage and sacrifices some computational efficiency.