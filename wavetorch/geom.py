from copy import deepcopy
from typing import Tuple
import numpy as np
import torch, os
from torch.nn.functional import conv2d
from .utils import load_file_by_type
from .utils import to_tensor


class WaveGeometry(torch.nn.Module):
    def __init__(self, domain_shape: Tuple, h: float, abs_N: int = 20, equation: str = "acoustic"):
        super().__init__()

        self.domain_shape = domain_shape

        self.register_buffer("h", to_tensor(h))

        self.register_buffer("abs_N", to_tensor(abs_N, dtype=torch.uint8))

        # INIT boundary coefficients
        self._init_b(abs_N)
        self._init_cpml(abs_N)
        # self.boundary = {"acoustic": self._init_b(abs_N), 
        #                  "elastic" : self._init_cpml(abs_N)}        
        
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
    
    @property
    def cmax(self):
        """Helper function for getting the maximum wave speed for calculating CFL"""
        #return np.max([self.c0.item(), self.c1.item()])
        return np.max(self.vp.detach().numpy())

    def _init_cpml(self, abs_N:int, f0:float=10.0, cp:float=1500., pa:float=1., pd:float=2., pb:float=1.):
        """Initialize the distribution of the d for unsplit PML"""
        """[1], Wei Zhang, Yang Shen, doi:10.1190/1.3463431"""
        self._init_d(abs_N, cp=cp, order=pd)

    def _corners(self, abs_N, d, dx, dy):
        Nx, Ny = self.domain_shape
        for j in range(Ny):
            for i in range(Nx):
                # Left-Top
                if i < abs_N+1 and j< abs_N+1:
                    if i < j:
                        d[i,j] = dy[i,j]
                    else:
                        d[i,j] = dx[i,j]
                # Left-Bottom
                if i > (Nx-abs_N-2) and j < abs_N+1:
                    if i + j < Nx:
                        d[i,j] = dx[i,j]
                    else:
                        d[i,j] = dy[i,j]
                # Right-Bottom
                if i > (Nx-abs_N-2) and j > (Ny-abs_N-2):
                    if i - j > Nx-Ny:
                        d[i,j] = dy[i,j]
                    else:
                        d[i,j] = dx[i,j]
                # Right-Top
                if i < abs_N+1 and j> (Ny-abs_N-2):
                    if i + j < Ny:
                        d[i,j] = dy[i,j]
                    else:
                        d[i,j] = dx[i,j]

    def _init_d(self, abs_N, order:float = 2.0, cp:float = 1500.):

        Nx, Ny = self.domain_shape

        R = 10**(-((np.log10(abs_N)-1)/np.log10(2))-3)
        #d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
        R = 1e-6; order = 2; cp = 3000.# Mao shibo Master
        d0 = (1.5*cp/abs_N)*np.log10(R**-1)
        d_vals = d0 * torch.linspace(0.0, 1.0, abs_N + 1) ** order
        d_vals = torch.flip(d_vals, [0])

        d_x = torch.zeros(Ny, Nx)
        d_y = torch.zeros(Ny, Nx)
        
        if abs_N > 0:
            d_x[0:abs_N + 1, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
            d_x[(Ny - abs_N - 1):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)

            d_y[:, 0:abs_N + 1] = d_vals.repeat(Ny, 1)
            d_y[:, (Nx - abs_N - 1):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

        self.register_buffer("_d", torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1))
        self._corners(abs_N, self._d, d_x.T, d_y.T)

    def _init_b(self, abs_N: int, B:float = 100.0, mode = 'cosine'):
        """Initialize the distribution of the damping parameter for the PML"""

        Nx, Ny = self.domain_shape

        assert Nx > 2 * abs_N + 1, "The domain isn't large enough in the x-direction to fit absorbing layer. Nx = {} and N = {}".format(
            Nx, abs_N)
        assert Ny > 2 * abs_N + 1, "The domain isn't large enough in the y-direction to fit absorbing layer. Ny = {} and N = {}".format(
            Ny, abs_N)
            
        b_x = torch.zeros(Ny, Nx)
        b_y = torch.zeros(Ny, Nx)
                
        if mode == 'cosine':
            idx = (torch.ones(abs_N + 1) * (abs_N+1)  - torch.linspace(0.0, (abs_N+1), abs_N + 1))/(2*(abs_N+1))
            b_vals = torch.cos(np.pi*idx)
            b_vals = torch.ones_like(b_vals) * B * (torch.ones_like(b_vals) - b_vals)

            b_x[0:abs_N+1,:] = b_vals.repeat(Nx, 1).transpose(0, 1)
            b_x[(Ny - abs_N - 1):Ny, :] = torch.flip(b_vals, [0]).repeat(Nx, 1).transpose(0, 1)
            b_y[:, 0:abs_N + 1] = b_vals.repeat(Ny, 1)
            b_y[:, (Nx - abs_N - 1):Nx] = torch.flip(b_vals, [0]).repeat(Ny, 1)

        self.register_buffer("_b", torch.sqrt(b_x ** 2 + b_y ** 2).transpose(0, 1))    

class WaveGeometryFreeForm(WaveGeometry):
    def __init__(self, **kwargs):

        h = kwargs['geom']['h']
        abs_N = kwargs['geom']['pml']['N']
        domain_shape = kwargs['domain_shape']
        self.autodiff = kwargs['autodiff']
        self.dt = kwargs['geom']['dt']
        self.device = kwargs['device']
        self.padding = abs_N
        self.source_type = kwargs['geom']['source_type']
        self.receiver_type = kwargs['geom']['receiver_type']
        self.save_interval = kwargs['training']['save_interval']
        self.model_parameters = []

        super().__init__(domain_shape, h, abs_N)

        self._init_model(kwargs['VEL_PATH'])
        # Determine the equation type elastic or acoustic
        self.equation = self.determine_eq_type()

        
    def _init_model(self, modelPath, shape = None, pml_width = None):
        for mname, mpath in modelPath.items():
            # If path is not None, read it and add to graph
            if mpath:
                assert os.path.exists(mpath), f"Cannot find model '{mpath}'"
                self.model_parameters.append(mname)
                self.__setattr__(mname, self.add_parameter(mpath))

        # vp = self.pad(load_file_by_type(modelPath['vp'], shape=shape, pml_width=pml_width))
        # self.model_vp = torch.nn.Parameter(to_tensor(vp))

    def determine_eq_type(self,):
        paras = self.model_parameters
        if len(paras)==1 and 'vp' in paras: return "acoustic"
        if len(paras)==3 and 'Q'  in paras: return "viscoacoustic"
        if len(paras)==3 and "vs" in paras: return "elastic"
    
    def __repr__(self):
        return f"Paramters of {self.model_paramters} have been defined."

    # Add the torch paramter
    def add_parameter(self, path, requires_grad=False):
        model = self.pad(load_file_by_type(path))
        return torch.nn.Parameter(to_tensor(model), requires_grad=True)

    def pad(self, d, mode='edge'):
        padding = self.padding
        return np.pad(d, ((padding, padding), (padding,padding)), mode=mode)
    
    @property
    def lame_lambda(self):
        return self.rho*(self.vp.pow(2)-2*self.vs.pow(2))

    @property
    def lame_mu(self):
        return self.rho*(self.vs.pow(2))

    # @property
    # def vp(self):
    #     return self.model_vp
    
    # @property
    # def vs(self):
    #     return self.model_vs
    
    # @property
    # def rho(self):
    #     return self.model_rho

    # @vp.setter
    # def vp(self, new_c):
    #     assert new_c.shape == self.domain_shape, "Shape of c must be {}, but got input shape {}".format(self.domain_shape, new_c.shape)
    #     if isinstance(new_c, torch.Tensor):
    #         self.vp = new_c
    #     if isinstance(new_c, np.ndarray):
    #         self.vp = torch.nn.Parameter(to_tensor(new_c).to(self.device))

    # @vs.setter
    # def vs(self, new_c):
    #     assert new_c.shape == self.domain_shape, "Shape of c must be {}, but got input shape {}".format(self.domain_shape, new_c.shape)
    #     if isinstance(new_c, torch.Tensor):
    #         self.vs = new_c
    #     if isinstance(new_c, np.ndarray):
    #         self.vs = torch.nn.Parameter(to_tensor(new_c).to(self.device))

    # @rho.setter
    # def rho(self, new_c):
    #     assert new_c.shape == self.domain_shape, "Shape of c must be {}, but got input shape {}".format(self.domain_shape, new_c.shape)
    #     if isinstance(new_c, torch.Tensor):
    #         self.rho = new_c
    #     if isinstance(new_c, np.ndarray):
    #         self.rho = torch.nn.Parameter(to_tensor(new_c).to(self.device))

    def save_model(self, path: str, paras: str, freq_idx=1, epoch=1):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.equation == 'elastic':
            _paras = {"vel":{
                            "vp": self.vp,
                            "vs": self.vs,
                            "rho": self.rho,},
                        "grad":{"vp": self.vp.grad,
                                "vs": self.vs.grad,
                                "rho":self.rho.grad,}
            }
        else:
            _paras = {"vel":{"vp": self.vp},
                     "grad":{"vp": self.vp.grad}}
            
        for para in paras:
            assert para in ["vel", "grad"], "para must be 'vel' or 'grad'"
            for key, data in _paras[para].items():
                save_path = os.path.join(path, f"{para}{key}F{freq_idx:02d}E{epoch:02d}.npy")
                if isinstance(data, np.ndarray):
                    np.save(save_path, data)
                elif isinstance(data, torch.Tensor):
                    np.save(save_path, data.cpu().detach().numpy())
            
    
        
