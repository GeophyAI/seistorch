import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F    

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
            tensors_for_meshgrid.append(torch.linspace(-1, 1, steps=size))
            # tensors_for_meshgrid.append(torch.linspace(0, size*dh/1000, steps=size))
        mgrid = torch.stack(torch.meshgrid(*tensors_for_meshgrid, indexing='ij'), dim=-1)
        mgrid = mgrid.reshape(-1, len(mshape))
        return mgrid

    def step(self,):
        pass

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords).view(self.domain_shape)
        return output, coords
    
class Encoder(nn.Module):

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
                 scale=(1,1),
                 dh=None):
        super().__init__()
        self.model_shape = domain_shape

        scaled_nz, scaled_nx = [int(x/s) for x,s in zip(self.model_shape, scale)]
        self.scaled_shape = (scaled_nz, scaled_nx)
        out_size_of_net = scaled_nz*scaled_nx

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_size_of_net)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        pretrained = '' if pretrained is None else pretrained
        if os.path.exists(pretrained):
            self.net.load_state_dict(torch.load(pretrained))
        else:
            print(f"Cannot find the pretrained model '{pretrained}'. Using random initialization.")

    def forward(self, latent_vector):
        output = self.net(latent_vector).view(self.scaled_shape)
        output = output.unsqueeze(0).unsqueeze(0)
        # print(output.shape)
        # output = F.interpolate(output, 
        #                   size=self.model_shape, 
        #                   mode='bilinear', 
        #                   align_corners=False)
        return output.squeeze(0).squeeze(0)


# class Encoder(nn.Module):
#     def __init__(self, 
#                  in_features,
#                  out_features,
#                  hidden_features, 
#                  hidden_layers, 
#                  domain_shape,
#                  scale=(10,10),
#                  **kwargs):
#         super(Encoder, self).__init__()
#         self.in_features = in_features
#         self.model_shape = domain_shape
#         # scale = (273, 661)
#         scaled_nz, scaled_nx = [int(x/s) for x,s in zip(domain_shape, scale)]
#         self.scaled_shape = (scaled_nz, scaled_nx)

#         # define input layer
#         self.input_layer = nn.Linear(in_features, hidden_features)

#         # self.out_features = out_features
#         out_size_of_net = scaled_nz*scaled_nx
#         self.output_size = torch.prod(torch.tensor(domain_shape)).item()

#         # Define hidden layers
#         self.hidden_layers = nn.ModuleList()
#         for _ in range(hidden_layers):
#             self.hidden_layers.append(nn.Linear(hidden_features, hidden_features))

#         # Define output layer
#         self.output_layer = nn.Linear(hidden_features, out_size_of_net)

#         # Define activation function
#         self.activation = nn.ReLU()
#         self.activation_out = nn.Sigmoid()

#         self.init_weights()

#     def init_weights(self):
#         with torch.no_grad():
#             self.input_layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
#             for layer in self.hidden_layers:
#                 layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
#             self.output_layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

#     def forward(self, x):
#         # Input layer
#         x = self.input_layer(x)
#         x = self.activation(x)

#         # hidden layers
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = self.activation(x)

#         # Output layer
#         x = self.output_layer(x)

#         # make show the output is > 0
#         # x = self.activation_out(x) DO NOT USE THIS

#         # if self.out_features != self.output_size:
#         #     # interpolate
#         # view for 1D to 2D
#         x = x.view(self.scaled_shape).unsqueeze(0).unsqueeze(0)
#         x = F.interpolate(x, 
#                           size=self.model_shape, 
#                           mode='bilinear', 
#                           align_corners=False)
#         return x.squeeze(0).squeeze(0)

        # For only 1 output
        # x = x*torch.ones(self.output_size).to(x.device)
        # x = x.reshape(*self.model_shape)

        # from (nz) to (nz, nx)
        # return x

class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, dropout_prob=0.25):
        super(ConvUpBlock, self).__init__()

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

        # Convolutional layer with leaky ReLU and dropout
        self.conv_leaky_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_leaky_relu(x)
        return x
    
class CNN(nn.Module):
    """Zhu et al. Geophysics, 2022. 10.1190/GEO2020-0933.1"""
    def __init__(self, 
                 in_features, 
                 out_features, 
                 hidden_features, 
                 hidden_layers, 
                 domain_shape, 
                 **kwargs):
        super(CNN, self).__init__()
        self.model_shape = domain_shape
        # Layer 1: Fully Connected layer with tanh activation
        self.layer1 = nn.Sequential(
            # nn.Linear(in_features, hidden_features),
            SineLayer(in_features, hidden_features, is_first=True, omega_0=30),
            nn.Tanh(),
        )

        # Intermediate layers using the UpsampleConvLeakyReLU module
        self.layers = nn.ModuleList([
            ConvUpBlock(1, 128),
            ConvUpBlock(128, 64),
            ConvUpBlock(64, 32),
            ConvUpBlock(32, 16)
        ])

        # Layer 6: 4x4 convolutional layer (1) + tanh
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=4, stride=1, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(1, 1, 2, 4)  # Reshape for convolutional layers

        # Iterate through intermediate layers
        for layer in self.layers:
            x = layer(x)
        # Final layer
        x = self.layer6(x)

        x = F.interpolate(x, 
                          size=self.model_shape, 
                          mode='bilinear', 
                          align_corners=False).squeeze(0).squeeze(0)

        return x