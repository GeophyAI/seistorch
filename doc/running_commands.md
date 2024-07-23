# Configuration Parameters for FWI Scripts

When running the Seistorch FWI scripts, such as `fwi.py`, `seistorch_dist.py` and `codingfwi.py`, you may need to configure additional parameters to tailor the simulations and inversions to your specific requirements. This section provides an overview of these essential configuration parameters and their roles in achieving successful FWI experiments.

## Parameters for `seistorch_dist.py`

### Parameter: [`config`]
- **Description**: Configuration file for geometry, training, and data preparation.
- **Example Usage**: `config.yml`

### Parameter: [`num_threads`] (Not supported now)
- **Description**: Sets the number of threads used for intraop parallelism on CPU (torch).
- **Example Usage**: `--num_threads 6` will use 6 threads for paralleism.

### Parameter: [`opt`]
- **Description**: Sepcify the optimizer for updating the model parameters.
- **Example Usage**: `--opt adam` will use Adam optimizer for updating the model parameters.
- **Note**: Only support Adam(`adam`), Steepest Descent(`sd`), Conjugate gradient(`cg`) optimizers.


### Parameter: [`loss`]
- **Description**: Sepcify the loss function for fitting the observed and synthetic data.
- **Example Usage**: `--loss vp=l2` will use MSE loss for inverting vp. Only support single loss function.

### Parameter: [`lr`]
- **Description**: Sepcify the learning rate for different model parameters.
- **Example Usage**: `--lr vp=10.0` will assign the initial learning rate of vp to `10.0`. This argument also supports multiple assignments, just as follows `--lr vp=10.0 vs=5.78 rho=1.25`.

### Parameter: [`grad-cut`]
- **Description**: Whether use mask to modify the gradient or not.
- **Example Usage**: `--grad-cut` will mask the gradient by the parameter `seabed` in configure file.

### Parameter: [`grad-smooth`]
- **Description**: Whether smooth the gradient or not.
- **Example Usage**: `--grad-smooth` will smooth the gradient by the parameters `smooth` in configure file.

### Parameter: [`grad-clip`]
- **Description**: Whether clip the gradient or not.
- **Example Usage**: `--grad-clip` will clip the gradient by the parameter `clipvalue` in sh file.

### Parameter: [`clipvalue`]
- **Description**: The value for clipping the gradient.
- **Default**: 0.02.
- **Example Usage**: `--clipvalue 0.02` will first calculate the q-th percentile of the gradient, and then clip the gradient by the value.

### Parameter: [`step-per-epoch`]
- **Description**: The number of steps per epoch.
- **Default**: 1.
- **Example Usage**: `--step-per-epoch 1` will update the model parameters once per loss backpropagation.

### Parameter: [`filteratfirst`]
- **Description**: Whether filter the wavelet before modeling.
- **Example Usage**: `--filteratfirst` will filter the wavelet before modeling.

### Parameter: [`obsnofilter`]
- **Description**: Whether filter the observed data.
- **Example Usage**: `--obsnofilter` will not filter the observed data.

## Parameters for `fwi.py`

### Parameter: [`config`]
- **Description**: Configuration file for geometry, training, and data preparation.
- **Example Usage**: `config.yml`

### Parameter: [`num_threads`]
- **Description**: Sets the number of threads used for intraop parallelism on CPU (torch).
- **Example Usage**: `--num_threads 6` will use 6 threads for paralleism.

### Parameter: [`num-batches`]
- **Description**: How many batches the data will be seperated. Seistorch will perform modeling `ceil(num-shots/num-batches)` times.
- **Example Usage**: `--num-batches 1`: All the data will be bundled in a single batch.

### Parameter: [`save-path`]
- **Description**: Sepcify the path for saving the inverted results.
- **Example Usage**: The arguments will overwrite the parameter `inv_savePath` in configure file. If not specified, the results will be saved in `inv_savePath`.

### Parameter: [`loss`]
- **Description**: Sepcify the loss function for fitting the observed and synthetic data.
- **Example Usage**: `--loss vp=l2` will use MSE loss for inverting vp. Only support single loss function.

### Parameter: [`lr`]
- **Description**: Sepcify the learning rate for different model parameters.
- **Example Usage**: `--lr vp=10.0` will assign the initial learning rate of vp to `10.0`. This argument also supports multiple assignments, just as follows `--lr vp=10.0 vs=5.78 rho=1.25`.

### Parameter: [`mode`]
- **Description**: Sepcify the mode of the module.
- **Example Usage**: Valid modes are `forward` and `inversion`.

### Parameter: [`grad-cut`]
- **Description**: Whether use mask to modify the gradient or not.
- **Example Usage**: `--grad-cut` will mask the gradient by the parameter `seabed` in configure file.

## Parameters for `codingfwi.py`

### Parameter: [`config`]
- **Description**: Configuration file for geometry, training, and data preparation.
- **Example Usage**: `config.yml`

### Parameter: [`num_threads`]
- **Description**: Sets the number of threads used for intraop parallelism on CPU (torch).
- **Example Usage**: `--num_threads 6` will use 6 threads for paralleism.

### Parameter: [`use-cuda`]
- **Description**: Use CUDA for acceelrating computations
- **Example Usage**: `--use-cuda` will assign the tasks on GPU.

### Parameter: [`gpuid`]
- **Description**: Sepcify the GPU id for using.
- **Example Usage**: `--gpuid 0` will assign the task on GPU 0.


### Parameter: [`loss`]
- **Description**: Sepcify the loss function for fitting the observed and synthetic data.
- **Example Usage**: `--loss vp=l2` will use MSE loss for inverting vp. If you want to use different loss function for different paramters, you can specify the arguments like this `--loss vp=l2 vs=l1`. Valid loss functions can be found in `seistorch/loss.py`

### Parameter: [`save-path`]
- **Description**: Sepcify the path for saving the inverted results.
- **Example Usage**: The arguments will overwrite the parameter `inv_savePath` in configure file. If not specified, the results will be saved in `inv_savePath`.

### Parameter: [`lr`]
- **Description**: Sepcify the learning rate for different model parameters.
- **Example Usage**: `--lr vp=10.0` will assign the initial learning rate of vp to `10.0`. This argument also supports multiple assignments, just as follows `--lr vp=10.0 vs=5.78 rho=1.25`.

### Parameter: [`batchsize`]
- **Description**: Sepcify the number of shots encoded in a super source.
- **Example Usage**: `--batchsize 20` will encoded 20 shots in a super source.

### Parameter: [`grad-smooth`]
- **Description**: Whether smooth the gradient or not.
- **Example Usage**: `--grad-smooth` will smooth the gradient before calling `optimizer.step()`.

### Parameter: [`grad-cut`]
- **Description**: Whether use mask to modify the gradient or not.
- **Example Usage**: `--grad-cut` will smooth the gradient by the parameter `seabed` in configure file.