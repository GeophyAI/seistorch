import numpy as np
import torch
from .operators import _laplacian
from .utils import to_tensor

def _time_step(b, c, y1, y2, dt, h):
    # Equation S8(S9)
    # When b=0, without boundary conditon.
    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                  (2 / dt**2 * y1 - torch.mul((dt**-2 - b * dt**-1), y2)
                   + torch.mul(c.pow(2), _laplacian(y1, h)))
                  )
    return y

class TimeStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, c, y1, y2, dt, h, t, it):
        ctx.models = c
        ctx.paras = [b, dt, h]
        ctx.save_gpu = it > 1
        if t%it == 0:
            ctx.save_for_backward(y1, y2)
        return _time_step(b, c, y1, y2, dt, h)

    @staticmethod
    def backward(ctx, grad_output):

        c = ctx.models
        b, dt, h = ctx.paras

        if ctx.save_gpu and (not ctx.saved_tensors):
            _o = torch.zeros_like(grad_output)
        elif (not ctx.save_gpu) or (ctx.save_gpu and ctx.saved_tensors):
            y1, y2 = ctx.saved_tensors

        grad_b = grad_c = grad_y1 = grad_y2 = grad_dt = grad_h = None

        if ctx.needs_input_grad[2]:
            # grad_y1 = ( dt.pow(2) * _laplacian(c.pow(2) *grad_output, h) + 2*grad_output) * (b*dt + 1).pow(-1)
            c2_grad = (b * dt + 1)**(-1) * c.pow(2) * grad_output
            grad_y1 = dt**(2) * _laplacian(c2_grad, h) + 2 * grad_output * (b * dt + 1).pow(-1)
        if ctx.needs_input_grad[3]:
            grad_y2 = (b * dt - 1) * (b * dt + 1).pow(-1) * grad_output

        if ctx.save_gpu and (not ctx.saved_tensors):
            return _o, _o, grad_y1, grad_y2, grad_dt, grad_h, None, None
        
        if ctx.needs_input_grad[0]:
            grad_b = - (dt * b + 1).pow(-2) * dt * (
                        c.pow(2) * dt**(2) * _laplacian(y1, h) + 2 * y1 - 2 * y2) * grad_output
        if ctx.needs_input_grad[1]:
            grad_c = (b * dt + 1).pow(-1) * (2 * c * dt**(2) * _laplacian(y1, h)) * grad_output

        return grad_b, grad_c, grad_y1, grad_y2, grad_dt, grad_h, None, None


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

    def forward(self, h1, h2, c_linear, rho, t, it):
        """Take a step through time

        Parameters
        ----------
        h1 : 
            Scalar wave field one time step ago (part of the hidden state)
        h2 : 
            Scalar wave field two time steps ago (part of the hidden state)
        """

        y = TimeStep.apply(self.geom.b, c_linear, h1, h2, self.dt, self.geom.h, t, it)

        return y, h1
