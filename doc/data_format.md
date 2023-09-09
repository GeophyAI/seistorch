# Data Format

In this chapter, we will introduce the format of the models used in Seistorch as well as the format of the data it generates.

## Model parameters

We recommend using the `.npy` file format to store your model files in Seistorch. This format is particularly efficient and convenient, especially for 2D/3D models, where the model shape should be `nz * nx`. Here are the advantages of using `.npy` files:


Here's an example of how to save and load a 2D seismic model using `.npy` format:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D model
model = np.random.rand(nz, nx)  # Replace with your actual model data
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

    ```python
    # Define source-receiver coordinates as a list of lists

    sources = [
        # Source 1
        [src_x1, src_z1],
        # Source 2
        [src_x2, src_z2]
    ]

    receivers = [
        # receiver of source 1:
        [[rec_x1, rec_x2, rec_x3], [rec_z1, rec_z2, rec_z3]],
        # receiver of source 2:
        [[rec_x1, rec_x2], [rec_z1, rec_z2]]
    ]


    write_pkl("source.pkl", sources)
    write_pkl("source.pkl", receivers)

    ```

    The length of the list `sources` and `receivers` muet be equal. So, if your receivers are fixed, you also need to specify them respectively. The purpose of doing this is to make it more fiexable to changing geomtry, such as OBC acquisition.

## Shot gather

The shot gather data is recorded in four dimensions `(nshots, nsamples, ntraces, ncomponents)`. The first dimension represents the number of shots, with each shot's data being in 3D. The second dimension corresponds to the number of time sampling points, the third dimension represents the number of traces (or receivers), and the fourth dimension represents the number of components.

Example: If the configures is set as follows and 20 receivers are used to record the wavefield, the shape of the shot gather is `(10, 2000, 20, 2)`.

```yaml
geom:
  receiver_type:
    - vx
    - vz

  nt: 2000
    
  Nshots: 10
```

**Note**: To handle cases where different shots may have a different number of receivers, we use `np.empty(Nshots, np.ndarray)` to create arrays for storing the shot gather data obtained during forward modeling. Therefore, when using `np.load` to load shot gather data, you should set `allow_pickle` to `True` to ensure proper loading.

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
