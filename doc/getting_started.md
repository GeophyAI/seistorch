# Getting Started

To start using Seistorch, follow these steps:

1. **Download the Source Code**: Begin by downloading the Seistorch source code from its GitHub repository. You can do this by cloning the repository to your local machine:

   ```shell
   git clone https://github.com/REPLACE_THIS_AFTER_UPLOADING.git

Seistorch provides two important scripts, `fwi.py` and `codingfwi.py`. `fwi.py` is used for running forward modeling and Classic Full-Waveform Inversion (FWI) tasks, respectively, and `codingfwi.py` for running source encoding-based fwi (only works in fixed-receivers settings yet).

## How to run forward modeling

Forward modeling in Seistorch allows you to simulate seismic wave propagation based on the following equation:

$$d=F(m, \mathbf{x}_r; \mathbf{x}_s, w(t))$$

Where:
- `m` represents the model parameters, describing subsurface properties.
- `xr` and `xs` are the receiver (detector) positions and source positions, respectively.
- `w(t)` represents the source wavelet or source signature.

1. **Prepare the configure file**:
Seistorch uses a YAML configuration file to specify the parameters required for forward modeling and Full-Waveform Inversion (FWI). The template for this configuration file can be found in the `template.yml` file provided with the source code. Please refers to section [**Configures**](configure.md) for details.

2. **Prepare the host file**: To prepare a host file, specify the host IP addresses and the number of GPUs to be used for running your application. Below is an example: 
    ```shell
    127.0.0.1:3
    ```
    which means you intend to run 3 processes on the local machine (127.0.0.1). One of these processes will be responsible for task assignment and I/O operations, while the other two will be used for computation. So, if you want to utilize 3 GPUs on a single node for your computations, you would need to specify a total of 4 processes.

    If you have two GPUs on node1 and four GPUs on node2, you should configure the host file as follows:

    Option1:
    ```shell
    node1:3
    node2:4
    ```

    Option2:
    ```shell
    node2:5
    node1:2
    ```

    When the number of processes specified in the host file exceeds the actual number of available GPU cards, all processes will be assigned to the available GPUs in a sequential manner.

3. **Run the command in shell**:

    **On GPU-SUPPORTED Machines**
    ```shell
    mpirun -f host python fwi.py configure.yml --mode forward --use-cuda
    ```
    **Using CPU Only**
    ```shell
    mpirun -f host python fwi.py configure.yml --mode forward
    ```
    In the above commands, both `fwi.py` and `configure.yml` must be specified as absolute paths.

## How to perform inversion
The goal of full waveform inversion is to recover subsurface properties $\mathbf m$ by minimizing the misfit between observed $\mathbf{d}^{obs}$ and synthetic seismic data $\mathbf{d}^{syn}$, which can be written as:
$$argmin_{\mathbf m} loss( \mathbf{d}^{obs}, \mathbf{d}^{syn})$$

The process of executing FWI is similar to the forward modeling, both requiring adjustments to the configuration file based on the model and the geometry.

1. **Prepare the configure file**:This step involves setting up the configure.yml file, which defines the parameters for both forward modeling and inversion. You can find detailed instructions on how to configure this file in the "## How to run forward modeling" section here.
2. **Prepare the host file**: (If you want to perform source-encoding fwi, just skip this procedure.) To specify the number of processes and the host configuration for your inversion, you should create a host file. This file will determine the distribution of processes across available GPUs or computing nodes. Refer to the instructions provided in the "## How to run forward modeling" section here for guidance on preparing the host file.

3. **Running command**:

    **Perform classis fwi**
    ```shell
    mpirun -f host \
    python fwi.py configure.yml  \
    --opt adam \
    --loss vp=l2 \
    --lr vp=10.0 \
    --mode inversion \
    --save-path /PARH_FOR_SAVING \
    --use-cuda \
    --grad-cut
    ```
    **Perform source-encoding fwi**
    ```shell
    python codingfwi.py configure.yml \
    --gpuid 0 \
    --opt adam \
    --loss vp=l2 \
    --mode inversion \
    --batchsize 20 \
    --lr vp=10.0 \
    --save-path /PATH_FOR_SAVING \
    --use-cuda \
    --grad-cut \
    --grad-smooth
    ```

All the inversion results will be saving at `save-path` the you specfied in the command.





