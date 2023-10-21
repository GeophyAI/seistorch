import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .eqconfigure import Parameters
from .utils import to_tensor
from .siren import Siren
from .io import SeisIO

class WaveGeometry(torch.nn.Module):
    def __init__(self, domain_shape: Tuple, h: float, abs_N: int = 20, equation: str = "acoustic", ndim: int = 2, multiple: bool = False):
        super().__init__()

        self.domain_shape = domain_shape

        self.multiple = multiple

        self.ndim = ndim

        self.register_buffer("h", to_tensor(h))

        self.register_buffer("abs_N", to_tensor(abs_N, dtype=torch.uint8))

        # INIT boundary coefficients
        generate_pml_coefficients = getattr(self, f"generate_pml_coefficients_{ndim}d")
        generate_pml_coefficients(abs_N)

    def state_reconstruction_args(self):
        return {"h": self.h.item(),
                "abs_N": self.abs_N.item()}

    def __repr__(self):
        return "WaveGeometry shape={}, h={}".format(self.domain_shape, self.h)

    def forward(self):
        raise NotImplementedError("WaveGeometry forward() is not implemented. " \
                                  "Although WaveGeometry is a subclass of a torch.nn.Module, its forward() method should never be called. " \
                                  "It only exists as a torch.nn.Module to hook into pytorch as a component of a WaveCell.")

    @property
    def c(self):
        raise NotImplementedError

    @property
    def b(self):
        return self._b

    @property
    def d(self,):
        return self._d

    def _corners(self, abs_N, d, dx, dy, multiple=False):
        Nx, Ny = self.domain_shape
        for j in range(Ny):
            for i in range(Nx):
                # Left-Top
                if not multiple:
                    if i < abs_N+1 and j< abs_N+1:
                        if i < j: d[i,j] = dy[i,j]
                        else: d[i,j] = dx[i,j]
                # Left-Bottom
                if i > (Nx-abs_N-2) and j < abs_N+1:
                    if i + j < Nx: d[i,j] = dx[i,j]
                    else: d[i,j] = dy[i,j]
                # Right-Bottom
                if i > (Nx-abs_N-2) and j > (Ny-abs_N-2):
                    if i - j > Nx-Ny: d[i,j] = dy[i,j]
                    else: d[i,j] = dx[i,j]
                # Right-Top
                if not multiple:
                    if i < abs_N+1 and j> (Ny-abs_N-2):
                        if i + j < Ny: d[i,j] = dy[i,j]
                        else: d[i,j] = dx[i,j]

    def generate_pml_coefficients_2d(self, N=50, B=100.):
        Nx, Ny = self.domain_shape

        R = 10**(-((np.log10(N)-1)/np.log10(2))-3)
        #d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
        R = 1e-6; order = 2; cp = 1000.# Mao shibo Master
        d0 = (1.5*cp/N)*np.log10(R**-1)
        d_vals = d0 * torch.linspace(0.0, 1.0, N + 1) ** order
        d_vals = torch.flip(d_vals, [0])

        d_x = torch.zeros(Ny, Nx)
        d_y = torch.zeros(Ny, Nx)
        
        if N > 0:
            d_x[0:N + 1, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
            d_x[(Ny - N - 1):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
            if not self.multiple:
                d_y[:, 0:N + 1] = d_vals.repeat(Ny, 1)
            d_y[:, (Nx - N - 1):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

        self.register_buffer("_d", torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1))
        self._corners(N, self._d, d_x.T, d_y.T, self.multiple)
        # np.save("/home/wangsw/inversion/2d/layer/results/l2/pml.npy", self._d.cpu().detach().numpy())

    def generate_pml_coefficients_3d(self, N=50, B=100.):
        nz, ny, nx = self.domain_shape
        # Cosine coefficients for pml
        idx = (torch.ones(N + 1) * (N+1)  - torch.linspace(0.0, (N+1), N + 1))/(2*(N+1))
        b_vals = torch.cos(torch.pi*idx)
        b_vals = torch.ones_like(b_vals) * B * (torch.ones_like(b_vals) - b_vals)

        b_x = torch.zeros((nz, ny, nx))
        b_y = torch.zeros((nz, ny, nx))
        b_z = torch.zeros((nz, ny, nx))

        b_x[:,0:N+1,:] = b_vals.repeat(nx, 1).transpose(0, 1)
        b_x[:,(ny - N - 1):ny,:] = torch.flip(b_vals, [0]).repeat(nx, 1).transpose(0, 1)

        b_y[:,:,0:N + 1] = b_vals.repeat(ny, 1)
        b_y[:,:,(nx - N - 1):nx] = torch.flip(b_vals, [0]).repeat(ny, 1)

        b_z[0:N + 1, :, :] = b_vals.view(-1, 1, 1).repeat(1, ny, nx)
        b_z[(nz - N - 1):nz + 1, :, :] = torch.flip(b_vals, [0]).view(-1, 1, 1).repeat(1, ny, nx)

        #pml_coefficients = torch.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2)

        self.register_buffer("_d", torch.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2))

class WaveGeometryFreeForm(WaveGeometry):
    def __init__(self, mode='forward', **kwargs):

        self.mode = mode
        self.autodiff = True
        h = kwargs['geom']['h']
        abs_N = kwargs['geom']['pml']['N']
        self.padding = kwargs['geom']['pml']['N']
        self.domain_shape = kwargs['domain_shape']
        self.dt = kwargs['geom']['dt']
        self.boundary_saving = kwargs['geom']['boundary_saving']
        self.device = kwargs['device']
        self.source_type = kwargs['geom']['source_type']
        self.receiver_type = kwargs['geom']['receiver_type']
        self.multiple = kwargs['geom']['multiple']
        self.model_parameters = []
        self.inversion = False
        self.kwargs = kwargs

        self.seismp = ModelProcess(kwargs)
        self.seisio = SeisIO(kwargs)

        self.ndim = 2 if kwargs['geom']['Nz'] == 0 else 3
        super().__init__(self.domain_shape, h, abs_N, ndim=self.ndim, multiple=kwargs['geom']['multiple'])
        self.equation = kwargs["equation"]
        self.use_implicit = kwargs["training"]['implicit']['use']
        # Initialize the model parameters if not using implicit neural network
        self._init_model(kwargs['VEL_PATH'], kwargs['geom']['invlist'])
        # Initialize the implicit neural network if using implicit neural network
        if self.use_implicit: self._init_siren()

    def _init_siren(self,):
        # inn stands for implicit neural network
        self.coords = self.get_mgrid_from_vel(self.domain_shape)
        self.siren = dict()
        for par in self.pars_need_invert:
            self.siren[par] = Siren(in_features=2, out_features=1, hidden_features=128,
                                    hidden_layers=4, outermost_linear=True)
            # load the pretrained model if it exists
            pretrained = self.kwargs['training']['implicit']['pretrained']
            pretrained = '' if pretrained is None else pretrained
            if os.path.exists(pretrained):
                self.siren[par].load_state_dict(torch.load(pretrained))
            else:
                print(f"Cannot find the pretrained model '{pretrained}'")
            # send the siren to the target device
            self.siren[par].to(self.device)

    def _init_model(self, modelPath: dict, invlist: dict):
        """Initilize the model parameters
        Args:
            modelPath (dict): The dictionary that contains the path of model files.
            invlist (dict): The dictionary that specify whether invert the model or not.
        """
        needed_model_paras = Parameters.valid_model_paras()[self.equation]
        self.true_models = dict()
        self.pars_need_invert = []
        for para in needed_model_paras:

            # add the model to the graph
            mname, mpath = para, modelPath[para]
            print(f"Loading model '{mpath}', invert = {invlist[mname]}")
            if invlist[mname]:
                self.pars_need_invert.append(mname)
            # add the model to a list for later use
            self.model_parameters.append(mname)
            # load the ground truth model for calculating the model error
            true_model_path = self.kwargs['geom']['truePath'][mname]
            if true_model_path is not None and os.path.exists(true_model_path):
                self.true_models[mname]=self.seisio.fromfile(true_model_path)
            # load the initial model for the inversion
            if not self.use_implicit:
                invert = False if self.mode=='forward' else invlist[mname]
                self.__setattr__(mname, self.seismp.add_parameter(mpath, invert))

        # Loop over all the model parameters (invert=True)
        # for par in self.pars_need_invert:
        #     # Adding support for implicit velocity model
        #     nz, nx = self.domain_shape
        #     model = self.siren(self.coords)[0].view(nz, nx)
        #     # an-ti normalization for getting the true values
        #     mean = 1000.
        #     std = 3000.
        #     model = model * std + mean
        #     self.__setattr__(par, model)

    def step_implicit(self, mask):
        coords = self.coords # x, y coordinates
        shape = self.domain_shape # shape of the model
        for par in self.pars_need_invert:
            par_value = self.siren[par](coords)[0].view(shape)
            if mask is not None:
                mask = torch.nn.functional.pad(mask, (self.padding,)*4, mode='constant', value=0)
            # an-ti normalization for getting the true values
            anti_value = self.anti_normalization(par_value)
            #anti_value *= mask
            anti_value[0:self.padding+12] = 1500.
            setattr(self, par, anti_value)

    def step(self, seabed):
        """
            Doing this step for each iteration.
        """
        # If we use implicit neural network for reparameterization, 
        # we need to reset model parameters by input the coords of 
        # the model to the implicit neural network.
        # e.g: vp = self.siren['vp'](coords); 
        # e.g: vs = self.siren['vs'](coords);
        if self.use_implicit:
            self.step_implicit(seabed)

    def anti_normalization(self, model, mean=3000., std=1000.):
        return model * std + mean
    
    def __repr__(self):
        return f"Paramters of {self.model_parameters} have been defined."
        
    def pad_model_with_random_values(self, model, N):
        nz, nx = model.shape

        # min_val = np.min(model)
        # max_val = np.max(model)
        min_val = 400
        max_val = np.min(model)

        padded_model = np.zeros((nz + 2 * N, nx + 2 * N))

        # Set the inner values
        padded_model[N:N+nz, N:N+nx] = model

        # Set the boundary values
        for i in range(N):
            padded_model[i, :] = np.random.uniform(min_val, max_val, (nx + 2 * N))  # 
            padded_model[-i-1, :] = np.random.uniform(min_val, max_val, (nx + 2 * N))  # 
            padded_model[:, i] = np.random.uniform(min_val, max_val, (nz + 2 * N))  # 
            padded_model[:, -i-1] = np.random.uniform(min_val, max_val, (nz + 2 * N))  # 

        return padded_model
    
    def tensor_to_img(self, key, array, padding=0, vmin=None, vmax=None, cmap="seismic"):
        cmap = plt.get_cmap(cmap)
        array = array[padding:-padding, padding:-padding]
        #
        if vmin is None:
            vmin = array.min()
        if vmax is None:
            vmax = array.max()
        
        array = np.clip(array, vmin, vmax)
        
        # Normalize to [0, 1]
        array = (array - vmin) / (vmax - vmin+1e-10)
        
        # Apply the colormap
        img = cmap(array)
        
        # Convert to numpy
        img = torch.from_numpy(img).permute(2, 0, 1)#.unsqueeze(0)

        return img
    
    def set_zero_boundaries(self, tensor, pad=50):
        tensor[..., :pad, :] = 0
        tensor[..., -pad:, :] = 0
        tensor[..., :, :pad] = 0
        tensor[..., :, -pad:] = 0
        return tensor

    def get_mgrid_from_vel(self, shape):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        nz, nx = shape
        xtensor = torch.linspace(-1, 1, steps=nx)
        ztensor = torch.linspace(-1, 1, steps=nz)
        mgrid = torch.stack(torch.meshgrid(ztensor, xtensor, indexing='ij'), dim=-1)
        mgrid = mgrid.reshape(-1, 2)
        return mgrid.to(self.device)

    def gradient_smooth(self, ):

        for para in self.model_parameters:
            var = self.__getattr__(para)
            if var.requires_grad:
                # copy data from tensor to numpy
                smoothed_grad = var.grad.cpu().detach().numpy()
                # smooth the data
                smoothed_grad = self.seismp.smooth(smoothed_grad)
                # copy data from numpy to tensor
                var.grad.copy_(to_tensor(smoothed_grad).to(var.grad.device))

    def gradient_cut(self, mask=None, padding=50):
        top = 0 if self.multiple else padding
        mask = torch.nn.functional.pad(mask, (padding, padding, top, padding), mode='constant', value=0)
        for para in self.model_parameters:
            var = self.__getattr__(para)
            if var.requires_grad:
                var.grad.data = var.grad.data * mask
    
    def gradient_clip(self,):
        grad_list = [param.grad for param in self.siren["vp"].parameters()]

        grad = torch.cat([grad.flatten() for grad in grad_list])

        bound = torch.quantile(grad, torch.Tensor([0.02, 0.98]).to(grad.device).to(grad.dtype))
        min_val = torch.min(bound[0])
        max_val = torch.max(bound[1])

        for para in self.siren["vp"].parameters():
            para.grad.data.clamp_(min=min_val, max=max_val)

    def reset_random_boundary(self,):
        for para in self.model_parameters:
            var = self.__getattr__(para).detach()
            if var.requires_grad:
                cut_var = var.cpu().detach().numpy()[self.padding:-self.padding, self.padding:-self.padding]
                pad_var = self.pad_model_with_random_values(cut_var, self.padding)
                var.copy_(to_tensor(pad_var).to(var.device))

    def save_model(self, path: str, paras: str, freq_idx=1, epoch=1, writer=None, max_epoch=1000):
        """Save the data of model parameters and their gradients(if they have).

        Args:
            path (str): The root save path.
            paras (str): not used.
            freq_idx (int, optional): The frequency index of multi scale. Defaults to 1.
            epoch (int, optional): The epoch of the current scale. Defaults to 1.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        for para in self.model_parameters: # para is in ["vp", "vs", "rho", "Q", ....]
            #var = self.__getattr__(para)
            var = getattr(self, para)
            if para not in self.pars_need_invert:
                continue
            # if the model parameter is in the invlist, then save it.
            var_par = var.cpu().detach().numpy()
            if not self.use_implicit:
                if var.grad is not None:
                    var_grad = var.grad.cpu().detach().numpy()
            else:
                var_grad = np.zeros_like(var_par)
            for key, data in zip(["para"+para, "grad"+para], [var_par, var_grad]):
                # Save the data of model parameters and their gradients(if they have) to disk.
                save_path = os.path.join(path, f"{key}F{freq_idx:02d}E{epoch:02d}.npy")
                np.save(save_path, data)

                # Calcualte the model error when the true model is known.
                if "para" in key and self.true_models:
                    _pad = self.padding
                    if self.multiple:
                        data_copy = data[:-_pad, _pad:-_pad]
                    else:
                        data_copy = data[_pad:-_pad, _pad:-_pad]
                    model_error = np.sum((data_copy - self.true_models[para])**2)

                # Write the data to tensorboard.
                if writer is not None:
                    tensor_data = self.tensor_to_img(key, data, padding=self.padding, vmin=None, vmax=None)
                    # Write the model parameters and their gradients(if they have) to tensorboard.
                    writer.add_images(key, 
                                        tensor_data, 
                                        global_step=freq_idx*max_epoch+epoch, 
                                        dataformats='CHW',)
                    # Write the model error to tensorboard.
                    writer.add_scalar(f"model_error/{para}", model_error, global_step=freq_idx*max_epoch+epoch)

class ModelProcess:

    def __init__(self, cfg, ):
        self.cfg = cfg
        self.io = SeisIO(cfg)

    # Add the torch paramter
    def add_parameter(self, path: str, requires_grad=False):
        """Read the model paramter and setting the attribute 'requires_grad'.

        Args:
            path (str): The path of the model file. 
            requires_grad (bool, optional): Wheter this parameter need to be inverted. Defaults to False.

        Returns:
            _type_: torch.nn.Tensor
        """
        model = self.pad(self.io.fromfile(path), mode="edge")
        return torch.nn.Parameter(to_tensor(model), requires_grad=requires_grad)

    def pad(self, d, mode="edge"):
        """Padding the model based on the PML width.

        Args:
            d (np.ndarray): The data need to be padded.
            mode (str, optional): padding mode. Defaults to 'edge'.

        Returns:
            np.ndarray: the data after padding.
        """
        mode_options = ['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap']
        assert mode in mode_options, f"mode must be one of {mode_options}"

        padding = self.cfg['geom']['pml']['N']
        multiple = self.cfg['geom']['multiple']
        ndim = 2 if self.cfg['geom']['Nz'] == 0 else 3

        top = 0 if multiple else padding
        if ndim==2:
            return np.pad(d, ((top, padding), (padding,padding)), mode=mode)
        elif ndim==3:
            _padding_ = ((padding, padding), )*ndim
            return np.pad(d, _padding_, mode=mode)
        
    def smooth(self, data):
        """Smooth the data.

        Args:
            data (np.ndarray): The data need to be smoothed.

        Returns:
            np.ndarray: The smoothed data.
        """
        
        smcfg = self.cfg['training']['smooth']

        counts = smcfg['counts']
        radius = (smcfg['radius']['z'], smcfg['radius']['x'])
        sigma = (smcfg['sigma']['z'], smcfg['sigma']['x'])

        smdata = data.copy()
        for _ in range(counts):
            smdata = gaussian_filter(smdata, sigma, radius=radius)
        return smdata
    
