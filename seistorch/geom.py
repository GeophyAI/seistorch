import os
import jax
import torch
import importlib

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .io import SeisIO
from .eqconfigure import Parameters
from .networks import Siren, Encoder, CNN, SirenScale
from .random import random_fill_2d
from .utils import to_tensor

from typing import Tuple
from scipy.ndimage import gaussian_filter


class WaveGeometry(torch.nn.Module):
    def __init__(self, 
                 domain_shape: Tuple, 
                 h: float, 
                 bwidth: int = 20, 
                 equation: str = "acoustic", 
                 ndim: int = 2, 
                 multiple: bool = False, 
                 sharding = None):
        
        super().__init__()

        self.domain_shape = domain_shape

        self.multiple = multiple
        self.ndim = ndim
        self.sharding = sharding
        
        if self.backend == 'torch':
            self.register_buffer("h", to_tensor(h))
        elif self.backend == 'jax':
            self.h = h

        # default boundary type is pml
        self.use_habc=False
        self.use_random=False
        self.use_random=False
        self.bwidth = bwidth
        if bwidth==0:
            self.register_buffer("_d", to_tensor(0.))

        self.setup_bc()


    def setup_bc(self,):

        btype = self.kwargs['geom']['boundary']['type']
        self.bwidth = self.kwargs['geom']['boundary']['width']

        # Choose the boundary type
        self.use_pml =  btype == 'pml' and self.bwidth > 0
        self.use_random = btype == 'random' and self.bwidth > 0
        self.use_habc = btype == 'habc' and self.bwidth > 0
        
        assert btype in ['pml', 'random', 'habc'], 'boundary type must be one of [pml, random, habc]'
        
        if self.use_pml or self.use_habc:

            module = importlib.import_module(f"seistorch.{btype}")
            coes_func = getattr(module, f"generate_{btype}_coefficients_{self.ndim}d", None)
            
            if btype =='habc':
                self.bwidth = 50
                if self.logger is not None:
                    self.logger.print(f"For HABC, width={self.bwidth} is fixed.")

            d = coes_func(self.domain_shape, self.bwidth, multiple=self.multiple)
            # np.save(f"{btype}.npy", d)
            if self.backend == 'torch':
                self.register_buffer("_d", to_tensor(d))
            elif self.backend == 'jax':
                self._d = jnp.array(d.cpu().numpy())
                if self.sharding is not None:
                    self._d = jax.device_put(self._d, self.sharding)

        if self.use_random:
            self.register_buffer("_d", to_tensor(0.))
            if self.logger is not None:
                self.logger.print(f"Using random boundary with width={self.bwidth}.")

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
    def __init__(self, mode='forward', logger=None, sharding=None, **kwargs):

        self.mode = mode
        self.kwargs = kwargs
        if 'unit' not in self.kwargs['geom'].keys():
            self.unit = 1.0
        else:
            self.unit = self.kwargs['geom']['unit']
        h = kwargs['geom']['h']*self.unit
        self.dt = kwargs['geom']['dt']
        self.nt = kwargs['geom']['nt']
        self.dh = kwargs['geom']['h']*self.unit
        self.device = kwargs['device']
        self.bwidth = kwargs['geom']['boundary']['width']
        self.domain_shape = kwargs['domain_shape']
        self.boundary_saving = kwargs['geom']['boundary_saving']
        self.source_type = kwargs['geom']['source_type']
        self.receiver_type = kwargs['geom']['receiver_type']
        self.multiple = kwargs['geom']['multiple']
        self.backend = kwargs['backend']
        self.spatial_order = kwargs['geom']['spatial_order']
        self.model_parameters = []
        self.inversion = False
        self.logger = logger

        self.seisio = SeisIO(kwargs)
        self.sharding = sharding

        if 'source_illumination' in kwargs['geom'].keys():
            self.source_illumination = kwargs['geom']['source_illumination']
        else:
            self.source_illumination = False

        # The demension of the model
        self.ndim = len(self.domain_shape)
        super().__init__(self.domain_shape, h, self.bwidth, ndim=self.ndim, multiple=kwargs['geom']['multiple'], sharding=self.sharding)
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
                self.__setattr__(mname, self.add_parameter(mpath, invert, self.unit, self.backend))
                
                if self.logger is not None:
                    self.logger.print(f"The max value of {mname} is {getattr(self, mname).max().item()}")
                    self.logger.print(f"The min value of {mname} is {getattr(self, mname).min().item()}")
    
    def __repr__(self):
        return f"Paramters of {self.model_parameters} have been defined."

    @property
    def padding_list(self,):
        top = 0 if self.multiple else self.bwidth

        if self.ndim == 2:
            return [[top, self.bwidth], [self.bwidth, self.bwidth]]
        
        if self.ndim == 3:
            return [[top, self.bwidth], 
                    [self.bwidth, self.bwidth], 
                    [self.bwidth, self.bwidth]]
    
    # Add the torch paramter
    def add_parameter(self, path: str, requires_grad=False, unit=1, backend='torch'):
        """Read the model paramter and setting the attribute 'requires_grad'.

        Args:
            path (str): The path of the model file. 
            requires_grad (bool, optional): Wheter this parameter need to be inverted. Defaults to False.

        Returns:
            _type_: torch.nn.Tensor
        """
        d = self.seisio.fromfile(path)*unit
        model = np.pad(d, self.padding_list, mode="edge")
        if self.logger is not None:
            self.logger.print(f"The model has been padded from {d.shape} to {model.shape}.")
        if backend == 'torch':
            return torch.nn.Parameter(to_tensor(model), requires_grad=requires_grad)
        else:
            return jnp.array(model)