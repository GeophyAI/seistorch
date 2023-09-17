# Quick Start
Welcome to Seistorch! This quick start guide will walk you through the basics of using Seistorch for seismic simulations and Full-Waveform Inversion (FWI). 

**Note**: The FWI results demonstrated in our example are approximate and indicate incomplete inversion. To achieve better inversion results, you may need to adjust various parameters, such as the learning rate, decay rates, source wavelet frequencies, and potentially other hyperparameters. Fine-tuning these parameters can significantly impact the quality and convergence of the inversion process, allowing you to obtain more accurate subsurface models. It often requires experimentation and tuning to find the optimal set of parameters for your specific seismic data and geological conditions.

We'll cover the following topics:

1. [**Running 2D Forward Modeling**](#2d-forward-modeling): Simulate seismic wave propagation in 2D space. Using a generated two layer 2D model as an example.

2. [**Running 3D Forward Modeling**](#3d-forward-modeling): Simulate seismic wave propagation in 3D space. Using a generated two layer 3D model as an example.

3. [**Boundary saving-based FWI**](#boundary-saving-based-automatic-differentiation): Perform Full-Waveform Inversion with boundary saving-based automatic differentiation. Using a generated two layer 2D model as an example.

4. [**Batched classic FWI**](#2d-batched-classic-fwi): How to bundle the shots into a batch to a batch for accelerating the computation of fwi. Using marmousi model as an example.

5. [**Source encoding acoustic FWI**](#2d-source-encoding-fwi): Perform source-encoding (phase and polarity encoding) based full waveform inversion. Using marmousi model as an example.

6. [**Source inversion**](#source-inversion): How to perform source inversion.


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

## Boundary saving-based automatic differentiation

The code of this section locates at `examples/ADvsBS`. This exmaples shows a workflow of performing full waveform inversion based on pure automatic differentiation (PAD) and boundary saving-based automatic differentiation (BSAD). The BSAD method is used to reduce the GPU memory usage by reconstructing the wavefield with boundary saving strategy during loss backpropagation.

- **Generate model and geometry**

    ```shell
    python generate_model_geometry.py
    ```

    The srcipt `generate_model_geometry.py` will generate a 2 layer ground truth model and a smoothed version of it. The corresponding source-receiver pairs will be generated as well. A figure named **model_geometry.png** illustrate the true and initial model.

    ![Geomtry](figures/ADvsBS/model_geometry.png)

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

    The BSAD method significantly reduces memory usage and sacrifices some computational efficiency. The gradients of the AD and BS methods are shown in the following figures.

    ![ADvsBS](figures/ADvsBS/compare_AD_BS.png)

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

## 2D Batched classic FWI

In classic FWI, the computation of seismic data for each shot is typically done in a serial or MPI-parallel manner, meaning each shot is computed individually. However, deep learning frameworks provide us with batch computation capabilities through APIs like `conv2d` and `conv3d`. This allows us to simultaneously compute multiple shots, improving computational efficiency.

Seistorch also provides batched computation (BC) functionality. The BC is only valid in classic fwi, because source-encoding-based fwi encoded several sources into a super-source.

When running `fwi.py` you can specify the number of batches into which all the shots will be distributed by setting the `num-batches` parameter. This allows you to control how the shots are grouped for processing. More details can be seen in [Running commands](running_commands.md).

The source code of this section locates at `examples/batched_inversion`. Please follow the following steps.

- **Generate geometry (OBN settings)**

    The acquisition of this examples is same as the **2D Source Encoding FWI**. A number of 93 sources and 461 receivers are used.

    ```shell
    python generate_model_geometry.py
    ```

    ![Model](figures/batched_inversion/model_geometry.png "Model")

- **Modeling observed data in a batched manner**

    In the `forward.sh`, we set `--num-batches 10`, it means that ten shot form a batch are computed simultaneously within a single process/on a single GPU. Since we need to calculate `93` shots and only use 1 GPU (the host file is set as: `127.0.0.1:2`), except for the last batch, which has a batch size of `3`, the batch size for all other batches is `10`.

- **Perform classic fwi**

    In classic fwi, you need to set the `minibatch` in `configure.yml` to `true`, and set a proper `batch_size` which is `10` in our case.

    In the `inversion.sh` script, we specify `num-batches` as `1`, which bundles ten shots into a single batch for gradient computation. This process is equivalent to separately computing the gradients for 10 shots and then summing them together.

    ```shell
    sh inversion.sh
    ```

- **Show the inverted results**

    The inverted resutls will be saved at `examples/batched_inversion/results/fwi_batched`. The following figure shows the final inverted result.

    ![Model](figures/batched_inversion/Inverted.png "Model")


## Source inversion
The code of this section locates at `examples/source_inversion`. This exmaples shows a workflow of performing source inversion based on pure automatic differentiation (PAD).

We first generate a two layer velocity model and a background velocity. A BSpline wavelet (Cao and Han, 2011) is used as the source for modeling the observed data. Running the following commands will generate the corresponding velocity models, geometry and wavelet for forward modeling.

```shell
python generate_model_geometry.py
```

The generated ground truth model and background model can be seen in the following figure. The wide band Bspline shown in the third column is used as the wavelet for modeling observed data.

![Geometry](figures/source_inversion/model_geometry.png)

Run the script `sh forward.sh` will generate the observed data.

```shell
sh forward.sh
```

The objective function in source inversion is the same as in Full Waveform Inversion (FWI), which aims to match observed data with synthetic data. However, the difference lies in FWI, where we set the model parameters as the optimization targets, while in source inversion, the source wavelet serves as the parameter to be optimized.

The script `source_inversion.py` provides a workflow of source inversion. A ricker wavelet is used as the initial source for optmization.

![Geometry](figures/source_inversion/true_initial_wavelet.png)

```shell
python source_inversion.py
```

The intermediate steps of the inversion process will be saved in the `results` folder.

The following figure shows the final inverted result. We can observe that while the maximum and minimum amplitudes may not be consistent, the phase of the waveform has been relatively well recovered.

![Geometry](figures/source_inversion/Final_result.png)

## How to calculate the adjoint source in torch?

The adjoint source is essentially the derivative of the objective function with respect to the synthetic seismic records. Therefore, to calculate the adjoint source, you only need to construct different loss functions and use `torch.autograd.grad` to build the computation graph for differentiation, i.e. `adj=torch.autograd.grad(loss(syn, obs), syn)`. 

If we use the l2 loss function (waveform amplitude loss), the adjoint is the element difference of the synthetic and observed data, i.e. `adj=2*(syn-obs)/syn.numel()`. More details can be seen from the source code `examples/cal_adjoint_source`.

While this process is implicit within automatic differentiation frameworks, you can use this approach to verify whether the adjoint sources are behaving as expected when you need to check their behavior. It provides a way to inspect and validate the correctness of the adjoint sources during the inversion process.

- **Model setup**

    1. Generate the acquisition geometry and velocity models.
    ```shell
    python generate_model_geometry.py
    ```

    2. Perform forward modeling to generate the observed data.
    ```shell
    sh forward.sh
    ```
    ![Model](figures/cal_adjoint_source/model_geometry.png "Model")

- **Using sesitroch to perform forward modeling**

    The script `cal_adjoint.py` shows a workflow of calculating the adjoint source by seistorch. In this script, we demonstrate the consistency between using PyTorch APIs and manually calculating the adjoint sources. The results indicate that they are consistent. 
    You can obtain the results by running the following script, and the results match the figure shown below.

    ```sh
    python cal_adjoint.py
    ```

    ![Model](figures/cal_adjoint_source/adjoint_source.png "Model")