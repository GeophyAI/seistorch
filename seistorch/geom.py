import os
from typing import Tuple
import importlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .eqconfigure import Parameters
from .utils import to_tensor
from .networks import Siren, Encoder, CNN, SirenScale
from .io import SeisIO
from .random import random_fill_2d

class WaveGeometry(torch.nn.Module):
    def __init__(self, 
                 domain_shape: Tuple, 
                 h: float, 
                 abs_N: int = 20, 
                 equation: str = "acoustic", 
                 ndim: int = 2, 
                 multiple: bool = False):
        
        super().__init__()

        self.domain_shape = domain_shape

        # self.multiple = multiple

        self.ndim = ndim

        self.register_buffer("h", to_tensor(h))

        self.register_buffer("abs_N", to_tensor(abs_N, dtype=torch.uint8))

        # default boundary type is pml
        use_pml=False
        self.use_random=False

        if abs_N==0:
            self.register_buffer("_d", to_tensor(0.))

        if 'boundary' not in self.kwargs['geom'].keys() and abs_N > 0:
            use_pml = True

        if 'boundary' in self.kwargs['geom'].keys():
            btype = self.kwargs['geom']['boundary']['type']
            bwidth = self.kwargs['geom']['boundary']['width']

            use_pml =  btype == 'pml' and bwidth > 0
            self.use_random = btype == 'random' and bwidth > 0
            
        if use_pml:
            module = importlib.import_module("seistorch.pml")
            coes_func = getattr(module, f"generate_pml_coefficients_{ndim}d", None)
            d = coes_func(domain_shape, abs_N, multiple=multiple)
            # np.save("pml.npy", d)

            self.register_buffer("_d", d)
            if self.logger is not None:
                self.logger.print(f"Using PML with width={abs_N}.")

        if self.use_random:
            self.register_buffer("_d", to_tensor(0.))
            if self.logger is not None:
                self.logger.print(f"Using random boundary with width={abs_N}.")

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

class WaveGeometryFreeForm(WaveGeometry):
    def __init__(self, mode='forward', logger=None, **kwargs):

        self.mode = mode
        self.autodiff = True
        self.kwargs = kwargs
        if 'unit' not in self.kwargs['geom'].keys():
            self.unit = 1.0
        else:
            self.unit = self.kwargs['geom']['unit']
        h = kwargs['geom']['h']*self.unit
        abs_N = kwargs['geom']['pml']['N']
        self.dh = kwargs['geom']['h']*self.unit
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
        self.logger = logger

        self.seismp = ModelProcess(kwargs)
        self.seisio = SeisIO(kwargs)

        self.ndim = 2 if kwargs['geom']['Nz'] == 0 else 3
        super().__init__(self.domain_shape, h, abs_N, ndim=self.ndim, multiple=kwargs['geom']['multiple'])
        self.equation = kwargs["equation"]
        self.use_implicit = kwargs["training"]['implicit']['use']
        # Initialize the model parameters if not using implicit neural network
        self._init_model(kwargs['VEL_PATH'], kwargs['geom']['invlist'])
        # Initialize the implicit neural network if using implicit neural network
        if 'type' not in kwargs['training']['implicit'].keys():
            self.nntype='siren'
        else:
            self.nntype = kwargs['training']['implicit']['type']
        
        if self.use_implicit: 
            self._init_nn()

    def _init_nn(self,):

        in_features=self.kwargs['training']['implicit']['in_features']
        out_features=self.kwargs['training']['implicit']['out_features']
        #out_features=torch.prod(torch.tensor(self.domain_shape)).item()
        hidden_features=self.kwargs['training']['implicit']['hidden_features']
        hidden_layers=self.kwargs['training']['implicit']['hidden_layers']
        self.nn = dict()
        # for siren
        # self.coords = self.get_mgrid_from_vel(self.domain_shape)
        if self.logger is not None:
            self.logger.print("Initilizing the implicit neural network ...")
            self.logger.print(f"nntype: {self.nntype}")
            self.logger.print(f"NN configures: {self.kwargs['training']['implicit']}")

        nn = {'siren': Siren, 
              'encoder': Encoder,
              'cnn': CNN, 
              'sirenscale': SirenScale}

        for par in self.pars_need_invert:
            self.nn[par] = nn[self.nntype](in_features=in_features,
                                           out_features=out_features,
                                           hidden_features=hidden_features,
                                           hidden_layers=hidden_layers,
                                           outermost_linear=True,
                                           domain_shape=self.domain_shape,
                                           pretrained=self.kwargs['training']['implicit']['pretrained'],
                                           scale=(10,20),
                                           dh=self.dh)
            self.nn[par].to(self.device)

        #self.rand_vec = 2*torch.randn(in_features, device=self.device)-1
        self.rand_vec = torch.Tensor([0.9,0.0,-0.9]).float().to(self.device)

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
            if self.logger is not None:
                self.logger.print(f"Loading model '{mpath}', invert = {invlist[mname]}")
            if invlist[mname]:
                self.pars_need_invert.append(mname)
            # add the model to a list for later use
            self.model_parameters.append(mname)
            # load the ground truth model for calculating the model error
            true_model_path = self.kwargs['geom']['truePath'][mname]
            if true_model_path is not None and os.path.exists(true_model_path):
                self.true_models[mname]=self.seisio.fromfile(true_model_path)*self.unit
            # load the initial model for the inversion
            if not self.use_implicit:
                invert = False if self.mode=='forward' else invlist[mname]
                self.__setattr__(mname, self.seismp.add_parameter(mpath, invert, self.unit))
                
                if self.logger is not None:
                    self.logger.print(f"The max value of {mname} is {getattr(self, mname).max().item()}")
                    self.logger.print(f"The min value of {mname} is {getattr(self, mname).min().item()}")
                
                # self.__setattr__(mname, self.seismp.add_parameter(mpath, invert))
                # getattr(self, mname).to(self.device)

    def step_implicit(self, mask):
        vmin, vmax = self.kwargs['training']['implicit']['vmin'], self.kwargs['training']['implicit']['vmax']
        for par in self.pars_need_invert:
            if self.nntype in ['siren', 'sirenscale']:
                coords = self.nn[par].coords.to(self.device) # x, y coordinates
                # coords = self.coords.to(self.device) # x, y coordinates
                par_value = self.nn[par](coords)[0]
            elif self.nntype=='encoder':
                par_value = self.nn[par](self.rand_vec)
            elif self.nntype=='cnn':
                par_value = self.nn[par](self.rand_vec)
            #if mask is not None:
                #mask = torch.nn.functional.pad(mask, (self.padding,)*4, mode='constant', value=0)
            # an-ti normalization for getting the true values
            # par_value = par_value.abs()

            # par_value.clamp_(min=0, max=vmax)
            # using std and mean

            anti_value = self.anti_normalization(par_value, 3000., 1000.)*self.unit
            print(anti_value.max(), anti_value.min())
            # using vmin and vmax
            # anti_value = (vmin+(vmax-vmin)*par_value)*self.unit
            # anti_value = par_value # no anti normalization
            #anti_value *= mask
            # anti_value[0:self.padding+1] = 1500.*self.unit
            setattr(self, par, anti_value)

    def step_random_boundary(self,):
        vmin, vmax = 600, 4600
        effective_length = 30 # in grid
        for par in self.model_parameters:
            var = self.__getattr__(par)
            # copy data from tensor to numpy
            var_data = var.cpu().detach().numpy()
            # var_data = var.clone().cpu().numpy()
            # reset the random boundary
            var_data = random_fill_2d(var_data, effective_length, self.padding, self.dh, self.dh, vmin, vmax)
            # np.save("vel_rand.npy", var_data)
            # copy data from numpy to tensor
            # tensor = torch.nn.Parameter(var_data)
            # setattr(self, par, tensor)

            var.copy_(to_tensor(var_data).to(var.device))

    def step(self, seabed=None):
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

        if self.use_random:
            self.logger.print("Resetting the random boundary ...")
            with torch.no_grad():
                self.step_random_boundary()
        
        # TODO: random boundary
        # 10.1190/geo2014-0542.1
        # use_random = self.kwargs['geom']['boundary']['type'] == 'random'
        # if use_random:
        #     for para in self.model_parameters:
        #         var = self.__getattr__(para)
        #         var.data = random_fill(var, self.padding, 400., var.max().item(), self.multiple)
        # pass

    def anti_normalization(self, model, mean=4000., std=1000.):
        return model * std + mean
    
    def __repr__(self):
        return f"Paramters of {self.model_parameters} have been defined."
    
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

    @property
    def padding_list(self,):
        top = 0 if self.multiple else self.padding
        return (top, self.padding, self.padding, self.padding)

    # @property
    # def unit(self,):
    #     # if 'unit' in self.kwargs['geom'].keys():
    #     #     self.unit = 1
    #     # else:
    #     #     self.unit = 0.001
    #     self.unit=0.001
    #     return self.unit

    def gradient_cut(self, mask=None, padding=50):
        top = 0 if self.multiple else padding
        if self.ndim==2: pads = (padding, padding, top, padding)
        if self.ndim==3: pads = (padding, padding, top, padding, padding, padding)
        mask = torch.nn.functional.pad(mask, pads, mode='constant', value=0)
        for para in self.model_parameters:
            # print(self.vp)
            var = getattr(self, para)#self.__getattr__(para)
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
                    data_copy = self.seismp.depad(data)
                    model_error = np.sum((data_copy - self.true_models[para])**2)

                # Write the data to tensorboard.
                if writer is not None:
                    if self.ndim==2:
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
    def add_parameter(self, path: str, requires_grad=False, unit=1):
        """Read the model paramter and setting the attribute 'requires_grad'.

        Args:
            path (str): The path of the model file. 
            requires_grad (bool, optional): Wheter this parameter need to be inverted. Defaults to False.

        Returns:
            _type_: torch.nn.Tensor
        """
        d = self.io.fromfile(path)*unit
        model = self.pad(d, mode="edge")
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
    
    def depad(self, d):
        """Depadding the model based on the PML width.

        Args:
            d (np.ndarray): The data need to be depadded.

        Returns:
            np.ndarray: the data after depadding.
        """
        padding = self.cfg['geom']['pml']['N']
        multiple = self.cfg['geom']['multiple']
        ndim = 2 if self.cfg['geom']['Nz'] == 0 else 3

        top = 0 if multiple else padding
        if ndim==2:
            return d[top:-padding, padding:-padding]
        elif ndim==3:
            return d[padding:-padding, padding:-padding, padding:-padding]
        
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
    
