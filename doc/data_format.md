# Data Format

In this chapter, we will introduce the format of the models used in Seistorch as well as the format of the data it generates.

## Model parameters

We recommend using the `.npy` file format to store your model files in Seistorch. This format is particularly efficient and convenient, especially for 2D/3D models, where the model shape should be `nz*nx` (2D) or `nx*nz*ny` (3D).


Here's an example of how to save and load a 2D/3D seismic model using `.npy` format:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D model
model = np.random.rand(nz, nx)  # Replace with your actual model data
plt.imshow(model)
plt.show()

# Create a 3D model
model = np.random.rand(nx, nz, ny)  # Replace with your actual model data
plt.imshow(model)
plt.show()

# Save the model to an .npy file
np.save('model.npy', model)

# Load the model from the .npy file
loaded_model = np.load('model.npy')

```

## Source and receivers list

The `sources` and `receivers` data are stored in `.pkl` (Python Pickle) files. These files allow you to conveniently save and load lists of sources and receivers. Here's how you can perform read and write operations with `.pkl` files:

- **How to read and save geometry:**
    ```python
    import pickle

    def write_pkl(path: str, data: list):
        # Open the file in binary mode and write the list using pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def read_pkl(path: str):
        # Open the file in binary mode and load the list using pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    ```


- **Generate your own source and receiver file:**

    In Seistorch, the coordinates of sources and receivers are typically stored as python `list`, and in most cases, one source may correspond to multiple receivers. This data structure can be represented as a list of lists, where each outer list item represents a source, and the inner list contains the coordinates of receivers associated with that source.

    ### 2D case

    The length of the sources list equal to the number of shots. Each element in the list is also a list with a length of 2, where the two elements in each inner list represent the x-coordinate (in grid) and z-coordinate (in grid) of a seismic source. 

    The length of the receivers list equal to the number of shots as well.

    **Note**: It's preferable for the coordinate values to be integers. If they are not integers, they will be truncated or rounded to the nearest whole number.

    ```python
    # Define source-receiver coordinates as a list of lists

    sources = [
        # Source 1
        [src_x1, src_z1],
        # Source 2
        [src_x2, src_z2]
    ]

    receivers = [
        # receivers of source 1:
        [[rec_x1, rec_x2, rec_x3], [rec_z1, rec_z2, rec_z3]],
        # receivers of source 2:
        [[rec_x1, rec_x2], [rec_z1, rec_z2]]
    ]


    write_pkl("sources.pkl", sources)
    write_pkl("receivers.pkl", receivers)

    ```

    ### 3D case

    The primary distinction between 3D and 2D coordinate systems is the presence of an additional y-coordinate in the 3D system.

    **Method 1**

    ```python
    # Define source-receiver coordinates as a list of lists

    sources = [
        # Source 1
        [src_x1, src_y1, src_z1],
        # Source 2
        [src_x2, src_y2, src_z2]
    ]

    receivers = [
        # receivers of source 1:
        [[rec_x1, rec_x2, rec_x3], [rec_y1, rec_y2, rec_y3], [rec_z1, rec_z2, rec_z3]],
        # receivers of source 2:
        [[rec_x1, rec_x2], [rec_y1, rec_y2], [rec_z1, rec_z2]]
    ]

    write_pkl("sources.pkl", sources)
    write_pkl("receivers.pkl", receivers)

    ```

    **Method2**

    In 3D cases, you can also specify the coordinates with a `np.ndarray`.

    ```python
    """Generate receiver list"""
    import numpy as np

    receivers = []
    xx = np.arange(0, 200, 2)
    yy = np.arange(0, 125, 5)
    grid = np.meshgrid(xx, yy)

    depth = np.zeros_like(grid[0])
    receivers.append([grid[0], grid[1], depth])
    ```

The length of the list `sources` and `receivers` muet be equal. So, if your receivers are fixed, you also need to specify them respectively. The purpose of doing this is to make it more fiexable to changing geomtry, such as towed acquisition.

## Shot gather

The shot gather data is recorded in four dimensions `(nshots, nsamples, ntraces, ncomponents)`. The first dimension represents the number of shots, with each shot's data being in 3D. The second dimension corresponds to the number of time sampling points, the third dimension represents the number of traces (or receivers), and the fourth dimension represents the number of components.

Example: If the configures is set as follows and 20 receivers are used to record the x and z component of velocity, the shape of the shot gather is `(10, 2000, 20, 2)`, which means the data has 10 shots, each trace has 2000 time samples and each shot has 20*2 traces.

```yaml
geom:
  receiver_type:
    - vx
    - vz

  nt: 2000
    
  Nshots: 10
```

**Note**: SUPPORTED FORMAT: `npy`, `hdf5`. 

- For `.npy` format, To handle cases where different shots may have a different number of receivers, we use `np.empty(Nshots, np.ndarray)` to create arrays for storing the shot gather data obtained during forward modeling. Therefore, for npy format, when using `np.load` to load shot gather data, you should set `allow_pickle` to `True` to ensure proper loading.

- For `.hdf5` format, the following codes can be used for creating a seistorch supported data:

    ```python
    with h5py.File(datapath, 'w') as f:
        for shot in range(nshots):
            f.create_dataset(f'shot_{shot}', (nt, nr, nc), dtype='f', chunks=True)
    ```

For more details, please refer to `seistorch/io.py`.

## Inverted results
The inverted results will be saved at the `save-path` which you specified in the commands. The file structure under the "save-path" folder is as follows:

```shell
save-path/
├─ .
├─ configure.yml
├── logs/
│ ├── .
│ └── tensorboard_log_files
├─ para{vp/vs/rho}F{02d}E{02d}.npy
├─ grad{vp/vs/rho}F{02d}E{02d}.npy
└─ loss.npy
```

- Configure file

    The "configure.yml" file stores all the parameters used for this inversion, and it can be used to reproduce the experiment. 

- Logging files

    The "logs" folder contains log files that can be displayed using TensorBoard. You can open and view them using the following command: `tensorboard --logdir save-path/logs`.

    The log files record various information, including loss data, model error information, CPU and GPU utilization rates, and more. These logs provide valuable insights and metrics related to the inversion process.


- Model related files

    The models and gradients obtained from the inversion are saved as files with names starting with `para` and `grad`, respecively. For example, the gradient and model files for the first frequency band and the 10th iteration of the S-wave velocity model would be named `paravsF00E09.npy` and `gradvsF00E09.npy`, respectively.

    **Note**: The saved "para" and "grad" files are both in numpy format and can be directly loaded using `np.load`. However, it's important to note that their shapes are not the same as those in the `truePath` or `initPath` files.For example, assuming the models in `truePath`" and `initPath` have a shape of `(nz, nx)` and there's a PML (Perfectly Matched Layer) thickness of `N`, when `multiple: true` is set, the inversion files have a shape of `(nz+N, nx+2N)`. This means that N layers are added as padding to the bottom, left, and right boundaries of the original model. When `multiple: false` is set, the inversion files have a shape of `(nz+2N, nx+2N)`.

- Loss file

    The `loss.npy` file records scalar loss values used in the inversion. It provides a summary of the optimization progress and how well the inversion process is minimizing the objective function.
