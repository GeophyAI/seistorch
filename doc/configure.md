# Configure file
The configure.yml file is a crucial component of your seismic inversion and forward modeling process. It contains essential parameter settings that determine how your seismic simulations and inversions are carried out. In this document, we will describe the key sections and parameters within the `configure.yml` file.

## Format of configure file
The `configure.yml` is formatted with `yaml`. Seistorch automatically parses the `configure.yml` configuration file into a python `dictionary`, making it easy to access and utilize the configuration parameters within your Seistorch scripts. You can read the `.yml` file by following codes:

```python
from yaml import CLoader as Loader
# Load the configure file
config_path = r'Your_YML_FILE_PATH.yml'
with open(config_path, 'r') as ymlfile:
    cfg = load(ymlfile, Loader=Loader)
```

## Parameters in Configure File

This section provides an overview of key parameters in the configuration file that are related to modeling and inversion.

### Path related parameters

Parameters related to file paths should be set as strings without the need for quotation marks.

Example:
```yaml
geom:
  obsPath: /home/data/obs.npy

  truePath:
    vp: /home/data/vp.npy
```

- `geom`->`obsPath`

  In **forward modeling**: The `geom` -> `obsPath` parameter specifies the path where the synthetic data generated during the simulation will be saved. 

  In **inversion**: The same `geom` -> `obsPath` parameter specifies the path to the observed seismic data that will be used as a reference for fitting.

- `geom`->`obsmask` and (`geom`->`synmask`)

  Only works in inversion. Without mask, the loss is calculated by `loss=criterion(syn, obs)`, with `*mask` parameter, the loss is computed by `loss=criterion(syn*synmask, obs*obsask)`. The shape of datamask must be the same as the shape of obs and syn. It's useful when performing wave-equation-based traveltime tomography or reverse time migration.

- `geom`->`truePath`

  In **forward modeling**: When running the `fwi.py` script in Seistorch and setting the `--mode` parameter to `forward`, the seistorch is configured to load model parameters from the `truePath` dictionary. This behavior is essential for conducting forward modeling simulations.

  In **inversion**: If `truePath` is specified during inversion, it will be utilized to calculate model errors and subsequently logged into the TensorBoard for record-keeping.

  The `Parameters` class in `seistorch/eqconfigure.py` defines the model parameters required by various equations and simulations. Each equation has its list of necessary parameters. If a required model parameter is missing or cannot be loaded, Seistorch will raise an error, indicating which equation encountered the issue.

- `geom`->`initPath`
  
  Only works in inversion. The parameter `initPath` refers to the *path* where the initial model is located.

- `geoms`->`sources`

  The `sources` parameter is a *path* that contains seismic source information and is used for both forward modeling and inversion processes.

- `geoms`->`receivers`

  The `receivers` parameter is a *path* that contains seismic source information and is used for both forward modeling and inversion processes.

- `geoms`->`seabed`

  Only works in inversion. The `seabed` parameter specifies information about the seabed. It has the same shape as the model parameters in `truePath` and `initPath`, with values of 0 in the seawater areas and 1 in other regions. The gradient in sea will be masked by `seabed`, which can be expressed as :

  $$loss.backward()\\
  model.grad = model.grad*seabed\\
  optimizer.step()
  $$

- `geoms`->`wavelet`

  The `wavelet` parameter specifies the *path* to the source wavelet and is effective in both forward modeling and inversion.

  **Note**: When this parameter is specified, the paramters `fm`, `wavelet_delay` will be ignored.

### Boolean parameters

For boolean parameters, valid values are:

  - `true`: Represents a positive or affirmative condition.
  - `false`: Represents a negative or negative condition.

Example:

```yaml
training:

  implicit:
    use: false

  minibatch: true
```

- `training`->`implicit`->`use`

  Only works in inversion. Use the implicit network (siren) for representing the model parameters or not.

- `training`->`minibatch` and `training`->`batch_size`

  Works in both classic fwi and source-encoding fwi. 
  
  In **classic fwi**, the `minibatch` parameter specifies how many individual (or batched) shots (equal to `batch_size`) are used to compute the gradient for one epoch. The shots are selected randomly.

  $$gradient=\Sigma^{batchsize}_{i} \partial loss(w(t, \mathbf {x}^i_s), \mathbf m)/\partial \mathbf m$$

  In **source encoding fwi**, the `minibatch` parameter specifies how many individual shots (equal to the `batch_size`) are encoded into a single super-shot for one forward modeling. This approach is used to accelerate the inversion process by reducing the computational cost.

  $$gradient= \partial loss(w(t, \Sigma^{batchsize}_{i} \mathbf {x}^i_s), \mathbf m)/\partial \mathbf m$$

- `geom`->`multiple`

  Works in both forward modeling and inversion.

  When `multiple` is set to `true`, the absorption boundary conditions on the upper boundary of the model will be deactivated. When set to `false`, absorption boundaries are applied on all sides of the model.

- `geom`->`boundary_saving`

  Only works in inversion. 
  
  When `boundary_saving` is set to `true`, during the `loss.backward()` operation, the boundary saving strategy is employed to reconstruct the wavefield for reducing the GPU memory usage. When set to `false`, pure automatic differentiation will be used.

- `geom`->`wavelet_inverse`

  Works in both forward modeling and inversion. This parameter refers to whether to invert (reverse) the polarity of the Ricker wavelet. 

- `invlist`

  Only works in inversion. The parameters in `invlist` specifies the parameters that need to be inverted or included in the inversion process.

  In multi-parameter inversion, it is possible to configure the inversion process to focus on optimizing or inverting specific parameters while keeping others fixed or constrained. This can be useful when you have a complex model with many parameters, but you are primarily interested in updating or estimating a subset of those parameters that are most relevant to your research or application.

  Example: 
  In elastic wave inversion, if you only want to invert for the P-wave velocity and S-wave velocity while keeping the density unchanged, you can configure it as follows.

  ```yaml
  equation: elastic
  geom:
    initPath:
      vp: ./velocity/init_vp.npy
      vs: ./velocity/init_vs.npy
      rho: ./velocity/init_rho.npy

    invlist:
      vp: true
      vs: true
      rho: false
  ```

### Restricted choice parameter

This type of parameters has a limited set of valid choices, causing errors if selections fall outside this predefined list.

- `dtype`

  The data type in torch, default is `float32`.

- `equation`

  The wave equation used for modeling. All implemented wave equations are stored in `seistorch/equations`. For example, when set to `equation: acoustic`, it will call the functions from the file with the same name, `acoustic.py` for both forward and inversion.

- `geom`->`source_type` and `geom`->`receiver_type`

  The `source_type` parameter specifies the type of seismic source, which determines the `wavefield` component where the source wavelet is loaded. This parameter allows you to control how the source wavelet influences the simulation.

  The `receiver_type` parameter specifies the type of seismic receiver, which determines the wavefield component that will be recorded and used for analysis.

  The `source_type` and `receiver_type` can be a list from the valid wavefield names of the *class* `Wavefield` in `seistorch/eqconfigure.py`

  Example: In elastic modeling, if you want to load the seismic source into the `txx` and `tzz` components, and record the velocity components, you can configure it as follows.

  ```yaml
  equation: elastic

  source_type:
    - txx
    - tzz

  receiver_type:
    - vx
    - vz
  ```

- `geom`->`boundary`->`type`

  Valid options are `habc` and `pml`. Please note that the `habc` only valid for second-order acoustic equations.


### Scalar parameters

- `seed`

    The `seed` is the seed value used in a random process and can be employed to reproduce an experiment or random outcome.

- `training`->`batch_size`

    See Boolean paramters `training`->`minibatch` and `training`->`batch_size`.

- `training`->`N_epochs`

    The number of epochs in each scale.

- `training`->`lr`

    The initial learning rate of the optimizer.

- `training`->`scale_decay` and `training`->`lr_decay`

    The parameter `scale_decay` represents the decay rate of the learning rate across different scales, while `lr_decay` refers to the exponential decay rate of the learning rate within each scale.

    In the $i_{th}$ epoch of the $n_{th}$ frequency band, the learning rate is given by:

    $$lr(n, i)=(lr\_init^{scale\_decay})^{lr\_decay}$$


- `training`->`filter_ord`

    The order of the filter. Recommand value: from 1 to 4.

- `training`->`smooth`

  `Seistorch` use `scipy.ndimage.gaussian_filter` to smooth the gradients before calling `optimizer.step()`.

  ```python
  smooth:
    counts: 10 # should be int, indictes the times for smooth
    radius: # radius of the gaussian kernel in x and z direction.
      x: 5
      z: 10
    sigma: # sigma of the gaussian kernel in x and z direction.
      x:
      z:
  ```

- `geom`->`wavelet_delay`

    The delay of the ricker wavelet.

- `geom`->`multiscale`

    The parameter `multiscale` specifies the dominant frequency for each scale in a multi-scale inversion. A low pass filter is used for each scale. A keyword `all` will use the original data and original wavelet for inversion.

    Example:

    ```yaml
    geom:
      multiscale:
        - - 1.0
          - 3.0
        - 5.0
        - all
    ```

- `geom`->`dt`

    The parameter `dt` represents the time interval or time step used in simulations, and it can also be the time interval between samples of a source wavelet.

- `geom`->`nt`

    The number of the time samples for simulation and recording.

- `geom`->`fm`

    The dominant frequency of the ricker wave. Only works when the paramter `wavelet` leaves blank.

- `geom`->`h`

    The grid size in both x and z directions.

- `geom`->`Nshots`

    The number of the shots for forward modeling. This parameter can be bigger than the length of the source list and receivers list defined in `geom`->`sources` and `geom`->`receivers`. 
    
    The actual number of shots simulated is determined as the minimum value between `Nshots` and the length of the `sources` list. This ensures that the number of simulated shots does not exceed the available seismic source data.

    $$actual_shots=min(Nshots, len(sources))$$

- `geom`->`boundary`->`bwidth`

    The width of the absorbing boundary conditions.



