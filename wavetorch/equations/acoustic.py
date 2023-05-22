import torch
from torch.nn.functional import conv2d

from .checkpoint_easy import checkpoint as ckpt
from .utils import to_tensor

def _laplacian(y, h):
    """Laplacian operator"""
    kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(y.device)
    operator = h ** (-2) * kernel
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=1).squeeze(1)

def _time_step(c, y1, y2, dt, h, b):
    # Equation S8(S9)
    # When b=0, without boundary conditon.
    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * y1 - torch.mul((dt**-2 - b * dt**-1), y2)
                + torch.mul(c.pow(2), _laplacian(y1, h)))
                )
    return y

class WaveCell(torch.nn.Module):
    """The recurrent neural network cell implementing the scalar wave equation"""

    def __init__(self, geometry):

        super().__init__()

        # Set values
        self.geom = geometry
        self.register_buffer("dt", to_tensor(self.geom.dt))

    def parameters(self, recursive=True):
        for param in self.geom.parameters():
            yield param

    def get_parameters(self, key=None, recursive=True):
        yield self.geom.__getattr__(key)

    def forward(self, wavefields, model_vars, **kwargs):
        
        """Take a step through time

        Parameters
        ----------
        h1 : 
            Scalar wave field one time step ago (part of the hidden state)
        h2 : 
            Scalar wave field two time steps ago (part of the hidden state)
        """

        save_condition=kwargs["is_last_frame"]

        checkpoint = self.geom.checkpoint
        forward = not self.geom.inversion
        inversion = self.geom.inversion
        geoms = [self.dt, self.geom.h, self.geom.d]

        if checkpoint and inversion:
            y = ckpt(_time_step, save_condition, *model_vars, *wavefields, *geoms)
        if forward or (inversion and not checkpoint):
            y = _time_step(*model_vars, *wavefields, *geoms)
        return y, wavefields[0]
