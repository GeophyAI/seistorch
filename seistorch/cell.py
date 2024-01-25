import inspect
import torch
from .utils import to_tensor
from .checkpoint_new import checkpoint as ckpt_acoustic
from .checkpoint import checkpoint as ckpt
from .habc import bound_mask

class WaveCell(torch.nn.Module):
    """The recurrent neural network cell implementing the scalar wave equation"""

    def __init__(self, geometry, forward_func=None, backward_func=None):

        super().__init__()

        # Set values
        self.geom = geometry
        self.register_buffer("dt", to_tensor(self.geom.dt))
        self.forward_func = forward_func
        self.backward_func = backward_func
        self.ckpt = ckpt_acoustic if 'acoustic' in inspect.getmodule(forward_func).__name__ else ckpt

    def setup_habc(self, batchsize):
        if self.geom.use_habc:
            self.habc_masks = bound_mask(*self.geom.domain_shape,
                                         self.geom.bwidth, 
                                         self.geom.device, 
                                         batchsize, 
                                         return_idx=True, 
                                         multiple=self.geom.multiple)

    def parameters(self, recursive=True):
        for param in self.geom.parameters():
            yield param

    def get_parameters(self, key=None, recursive=True, implicit=False):
        if implicit:
            for param in self.geom.nn[key].parameters():
                yield param
        else:
            yield getattr(self.geom, key)

    def forward(self, wavefields, model_vars, **kwargs):
        """Take a step through time
        Parameters
        ----------
        vx, vz, txx, tzz, txz : 
             wave field one time step ago
        vp  :
            Vp velocity.
        vs  :
            Vs velocity.
        rho : 
            Projected density, required for nonlinear response (this gets passed in to avoid generating it on each time step, saving memory for backprop)
        """
        save_condition=kwargs["is_last_frame"]
        source_term = kwargs["source"]

        using_boundary_saving = self.geom.boundary_saving
        forward = not self.geom.inversion
        inversion = self.geom.inversion
        geoms = self.dt, self.geom.h, self.geom.d
        habcs = self.habc_masks if self.geom.use_habc else None

        if using_boundary_saving and inversion:
            hidden = self.ckpt(self.forward_func, self.backward_func, source_term, save_condition, len(model_vars), *model_vars, *wavefields, *geoms, habcs=habcs)
        if forward or (inversion and not using_boundary_saving):
            hidden = self.forward_func(*model_vars, *wavefields, *geoms, habcs=habcs)

        return hidden