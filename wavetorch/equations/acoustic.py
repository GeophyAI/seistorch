import torch
from torch.nn.functional import conv2d
from .utils import restore_boundaries

from .utils import to_tensor

NPML = 50
N = 2

def _laplacian(y, h):
    """Laplacian operator"""
    kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(y.device)
    operator = h ** (-2) * kernel
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=1).squeeze(1)

def _time_step(*args):

    c = args[0]
    y1, y2 = args[1:3]
    dt, h, b = args[3:6]

    # When b=0, without boundary conditon.
    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * y1 - torch.mul((dt**-2 - b * dt**-1), y2)
                + torch.mul(c.pow(2), _laplacian(y1, h)))
                )
    return y, y1

def _time_step_backward(*args):

    vp = args[0]
    h1, h2 = args[1:3]
    dt, h, b = args[3:6]
    h_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    b = b.unsqueeze(0)

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    h1_copy, h2_copy = h1.clone(), h2.clone()

    for s_type in src_type:
        source_var = eval(s_type)
        source_var = src_func(source_var, src_values, -1)

    # Replace the original tensors with their sub-tensors within the computation region
    vp = vp[compute_region_slice]
    b = b[compute_region_slice]

    laplacian_ = _laplacian(h1, h)#[compute_region_slice]

    h1_copy = h1_copy[compute_region_slice]
    h2_copy = h2_copy[compute_region_slice]

    # When b=0, without boundary conditon.
    # h = torch.mul((dt**-2 + b * dt**-1).pow(-1),
    #             (2 / dt**2 * h1_copy - torch.mul((dt**-2 - b * dt**-1), h2_copy)
    #             + torch.mul(vp.pow(2), laplacian_))
    #             )
    h = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * h1 - torch.mul((dt**-2 - b * dt**-1), h2)
                + torch.mul(vp.pow(2), laplacian_))
                )

    # Write back the results to the original tensors, but only within the update region
    # h[update_region_slice] = h[(slice(None), slice(N,-N), slice(N,-N))]
    
    h = restore_boundaries(h, h_bd, NPML, N)


    return h, h1

# class WaveCell(torch.nn.Module):
#     """The recurrent neural network cell implementing the scalar wave equation"""

#     def __init__(self, geometry):

#         super().__init__()

#         # Set values
#         self.geom = geometry
#         self.register_buffer("dt", to_tensor(self.geom.dt))

#     def parameters(self, recursive=True):
#         for param in self.geom.parameters():
#             yield param

#     def get_parameters(self, key=None, recursive=True):
#         yield self.geom.__getattr__(key)

#     def forward(self, wavefields, model_vars, **kwargs):
        
#         """Take a step through time

#         Parameters
#         ----------
#         h1 : 
#             Scalar wave field one time step ago (part of the hidden state)
#         h2 : 
#             Scalar wave field two time steps ago (part of the hidden state)
#         """

#         save_condition=kwargs["is_last_frame"]

#         checkpoint = self.geom.checkpoint
#         forward = not self.geom.inversion
#         inversion = self.geom.inversion
#         geoms = [self.dt, self.geom.h, self.geom.d]

#         if checkpoint and inversion:
#             y = ckpt(_time_step, save_condition, *model_vars, *wavefields, *geoms)
#         if forward or (inversion and not checkpoint):
#             y = _time_step(*model_vars, *wavefields, *geoms)
#         return y, wavefields[0]
