# Quick Start
Welcome to Seistorch! This quick start guide will walk you through the basics of using Seistorch for seismic simulations and Full-Waveform Inversion (FWI). We'll cover the following topics:

1. **Running 2D Forward Modeling**: Simulate seismic wave propagation in 2D space.

2. **Running 3D Forward Modeling**: Simulate seismic wave propagation in 3D space.

3. **Classic acoustic FWI**: Perform Full-Waveform Inversion.

4. **Source encoding acoustic FWI**: Perform source-encoding (phase and polarity encoding) based full waveform inversion.

5. **Source inversion**: How to perform source inversion.

## 2d forward modeling

The code of this section locates at `examples/forward_modeling2d`. This example shows how to run forward modeling with your own model and geometry.

-   First we need to change the directory into it.

    ```shell
    cd examples/forward_modeling
    ```

-   Generate a two layer model and the corresponding sources-receivers pairs by typing:

    ```shell
    python generate_model_geometry.py
    ```

    Two new folders **geometry** and **velocity_model** will be created. The figure **model_geometry.png** illustrates the generated layer model and the locations of source-receiver pairs.

    ![Geometry](figures/forward_modeling2d/model_geometry.png)

-   Running the shell script `forward.sh`, a file called `shot_gather.npy` will be created.

    ```shell
    sh forward.sh
    ```

-   Show the shot gathers.

    ```shell
    python show_shotgather.py
    ```

    The plotted results will be save in **shot_gather.png** (shown as follows).
    
    ![ShotGather](figures/forward_modeling2d/shot_gather.png)

## 3d forward modeling

The code of this section locates at `examples/forward_modeling3d`. This example shows how to run forward modeling with your own model and geometry.

- **Generate geometry and velocity model**

    The script `generate_model_geometry.py` generates a 3D velocity model with two layers. A ricker source at the center of the model suface is used for modeling. Moreover, we have created a three-dimensional observational system, and a schematic diagram of this observational system will be generated after running this script.

    ```shell
    python generate_model_geometry.py
    ```

    ![Geometry](figures/forward_modeling3d/model_geometry.png)

- **Run forward modeling**

    Perform simulation by running script `forward.sh`.

    ```shell
    sh forward.sh
    ```

- **Show results**

    The modeled data has 1 shot with 2000 time samples, 128 traces and a single component (displacement in scalar wave equation). The first 64 and last 64 traces are recorded along different line directions. Run the script will show the recorded data.

    ```shell
    python show_shotgather.py
    ```
    ![Geometry](figures/forward_modeling3d/shot_gather.png)


If you wanna generate your own 3D geometry and 3D velocity model, please refer to the section [data format](data_format.md).

## 2D Acoustic full waveform inversion

The code of this section locates at `examples/acoustic_fwi2d`. This exmaples shows a workflow of performing full waveform inversion based on pure automatic differentiation (PAD) and boundary saving-based automatic differentiation (BSAD). The BSAD method is used to reduce the GPU memory usage by reconstructing the wavefield with boundary saving strategy during loss backpropagation.

- **Generate model and geometry**

    ```shell
    python generate_model_geometry.py
    ```

    The srcipt `generate_model_geometry.py` will generate a 2 layer ground truth model and a smoothed version of it. The corresponding source-receiver pairs will be generated as well. A figure named **model_geometry.png** illustrate the true and initial model.

    ![Geomtry](figures/acoustic_fwi2d/model_geometry.png)

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

    ![ADvsBS](figures/acoustic_fwi2d/compare_AD_BS.png)

## 2D Source Encoding FWI

This chapter primarily focuses on how to perform Seistorch's Source Encoding Full Waveform Inversion (FWI) using the same parameter file as Classic FWI. The difference lies in the utilization of `codingfwi.py` to perform the FWI process with source encoding.

- **Download the velocity model**

    The velocity model we used here is modified from marmousi1. We pad the marmousi1 model at left and right with 50 grids (1km) for better illuminating. You need to download the velocity model from our huggingface repo [seismic inversion](https://huggingface.co/datasets/shaowinw/seismic_inversion/tree/main/marmousi_customer/marmousi_20m) or [model scope repo](https://modelscope.cn/datasets/shaowinw/seismic_inversion/files). Perhaps you just need to run the script `download_vel.sh`.The ground truth model `true_vp.npy` and initial model `linear_vp.npy` are needed to run this example.

    The downloaded two model files should be saved in `./velocity_model`. 

- **Generate the acquisition**
    Once you have done the aboving steps, run the script the `generate_model_geometry.py`. Just like the other examples, it will generate `sources.pkl` and `receivers.pkl` which describes the acquisition of modeling.

    ```shell
    python generate_model_geometry.py
    ```

    ![Model](figures/source_encoding_fwi/model_geometry.png "Model")

- **Running forward modeling**

    Run the script `forward.sh` to generate the observed data.

    ```shell
    sh forward.sh
    ```
    ![ShotGather](figures/source_encoding_fwi/shot_gather.png "Model")

- **Running inversion**

    The same configure file `forward.yml` is used for both forward modeling and inversion. The arguments of source encoding can be found in `source_encoding_fwi.sh`. The meaning of the arguments of `codingfwi.py` can be found in [running commands](running_commands.md).

    ```shell
    sh source_encoding_fwi.sh
    ```

- **Show inverted results**

    The script `show_results.py` can be used to show the inverted results when the `source_encoding_fwi.sh` has been executed done.

    ```shell
    python show_results.py
    ```

    ![Inverted](figures/source_encoding_fwi/Inverted.jpg "Results")

## Source inversion
The code of this section locates at `examples/source_inversion`. This exmaples shows a workflow of performing source inversion based on pure automatic differentiation (PAD).

We first generate a two layer velocity model and a background velocity. A BSpline wavelet is used as the source for modeling the observed data. Running the following commands will generate the corresponding velocity models, geometry and wavelet for forward modeling.

```shell
python generate_model_geometry.py
```

Run the script `sh forward.sh` will generate the observed data.

```shell
sh forward.sh
```

The objective function in source inversion is the same as in Full Waveform Inversion (FWI), which aims to match observed data with synthetic data. However, the difference lies in FWI, where we set the model parameters as the optimization targets, while in source inversion, the source wavelet serves as the parameter to be optimized.

The script `source_inversion.py` provides a workflow of source inversion. A ricker wavelet is used as the initial source for optmization.

```shell
python source_inversion.py
```

The intermediate steps of the inversion process will be saved in the `results` folder.