import os
import torch
import numpy as np
from torch import nn

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class Siren(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 outermost_linear=False, 
                 pretrained=None,
                 first_omega_0=30, 
                 hidden_omega_0=30., 
                 domain_shape=None, 
                 dh=None):
        super().__init__()
        self.domain_shape = domain_shape
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        self.coords = self.generate_mesh(domain_shape, dh)
        self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        pretrained = '' if pretrained is None else pretrained
        if os.path.exists(pretrained):
            self.net.load_state_dict(torch.load(pretrained))
        else:
            print(f"Cannot find the pretrained model '{pretrained}'. Using random initialization.")
    
    def generate_mesh(self, mshape, dh):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        tensors_for_meshgrid = []
        for size in mshape:
            tensors_for_meshgrid.append(torch.linspace(0, size*dh, steps=size))
        mgrid = torch.stack(torch.meshgrid(*tensors_for_meshgrid), dim=-1)
        mgrid = mgrid.reshape(-1, len(mshape))
        return mgrid

    def step(self,):
        pass

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords).view(self.domain_shape)
        return output, coords