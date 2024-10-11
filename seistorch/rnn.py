from typing import Iterator, Tuple
import numpy as np
import jax.numpy as jnp
from jax import lax
import jax
import torch
from torch.nn.parameter import Parameter

from .eqconfigure import Wavefield
from .source import WaveSourceTorch
from .probe import WaveProbeTorch, WaveProbeJax
from .array import TensorList
from .eqconfigure import Parameters

class WaveRNNBase:
    def __init__(self, cell, source_encoding=False, backend='torch', sharding=None):
        self.cell = cell
        self.source_encoding = source_encoding
        self.backend = backend
        self.device = {'torch': cell.geom.device, 'jax': None}[backend]
        self.zeros = {'torch': torch.zeros, 'jax': jnp.zeros}[backend]
        self.arange = {'torch': torch.arange, 'jax': jnp.arange}[backend]
        self.wavefield_names = Wavefield(self.cell.geom.equation).wavefields
        self.sharding = sharding

    def initialize(self, nshots):

        batchsize = 1 if self.source_encoding else nshots
        self.nshots = batchsize
        self.hidden_state_shape = (batchsize,) + self.cell.geom.domain_shape
        # Set wavefields
        self.wavefield_names = Wavefield(self.cell.geom.equation).wavefields
        for name in self.wavefield_names:
            setattr(self, name, self.zeros(self.hidden_state_shape, device=self.device))

class WaveRNN(WaveRNNBase, torch.nn.Module):
    def __init__(self, cell, source_encoding=False):

        torch.nn.Module.__init__(self)
        WaveRNNBase.__init__(self, cell, source_encoding, 'torch', sharding=None)

        #  Check the availability of the type of sources and probes.
        self.use_implicit = self.cell.geom.use_implicit
        self.second_order_equation = self.cell.geom.equation in Parameters.secondorder_equations()
        self.source_illumination = self.cell.geom.source_illumination
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        if self.cell.geom.use_implicit:
            for key in self.cell.geom.nn:
                for name, param in self.cell.geom.nn[key].named_parameters(prefix, recurse, remove_duplicate):
                    yield name, param
        else:
            for name, param in self.cell.geom.named_parameters(prefix, recurse, remove_duplicate):
                yield name, param

    """Original implementation"""
    def forward(self, x, omega=10.0, super_source=None, super_probes=None, vp=None):
        """Propagate forward in time for the length of the inputs
        Parameters
        ----------
        x :
            Input sequence(s), batched in first dimension
        output_fields :
            Override flag for probe output (to get fields)
        """
        # Hacky way of figuring out if we're on the GPU from inside the model
        device = self.cell.geom.device
        # Init hidden states
        self.initialize(super_source.x.size(0))

        # Initialize habc if needed
        if self.cell.geom.use_habc: self.cell.setup_habc(self.nshots)

        if self.source_illumination:
            self.precondition = torch.zeros(self.cell.geom.domain_shape, device=device)

        recs = dict()
        y = TensorList()
        for key in self.cell.geom.receiver_type:
            recs[key] = []

        # Set model parameters
        if not self.use_implicit:
            for name in self.cell.geom.model_parameters:
                setattr(self, name, getattr(self.cell.geom, name))
        else:
            # THIS IS THE IMPLICIT VERSION
            # WILL BE DEPRECATED
            setattr(self, 'vp', vp)

        # Pack parameters
        model_paras = [getattr(self, name) for name in self.cell.geom.model_parameters]

        # Loop through time
        x = x.to(device)
        # Get the super source and super probes

        # Add source mask
        super_source.smask = torch.zeros(self.hidden_state_shape, device=device)

        bidx = torch.arange(self.nshots, device=device) if not self.source_encoding else torch.zeros(self.nshots, device=device, dtype=torch.int32)

        if hasattr(super_source, 'z'):  # 3D
            super_source.smask[bidx, super_source.z, super_source.y, super_source.x] = 1.0
        else:  # 2D
            super_source.smask[bidx, super_source.y, super_source.x] = 1.0

        reccounts = super_probes.reccounts if not self.source_encoding else super_probes.x.size(0)

        super_source.source_encoding = self.source_encoding
        super_source.second_order_equation = self.second_order_equation

        time_offset = 3 if self.second_order_equation else 0
        batched_records = []
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            
            # Propagate the fields
            wavefield = [getattr(self, name) for name in self.wavefield_names]
            tmpi = min(i+time_offset, x.shape[1]-1)
            wavefield = self.cell(wavefield, 
                                  model_paras, 
                                  is_last_frame=(i==x.size(1)-1), 
                                  omega=omega, 
                                  source=[self.cell.geom.source_type, super_source, x[..., tmpi].view(xi.size(1), -1)])
            
            # if True:
                # np.save(f"./wf_pml/wf_foward{i:04d}.npy", wavefield[0].detach().cpu().numpy())

            # Set the data to vars
            for name, data in zip(self.wavefield_names, wavefield):
                setattr(self, name, data)

            # Add source
            for source_type in self.cell.geom.source_type:
                setattr(self, source_type, super_source(getattr(self, source_type), xi.view(xi.size(1), -1)))

            # Measure probe(s): new implementation
            for key in self.cell.geom.receiver_type:
                recs[key].append(super_probes(getattr(self, key)))

            if self.source_illumination:
                self.precondition += torch.sum(getattr(self, source_type).detach()**2, 0)
            
        # stacked_data: (nbatches, nt, nreceivers all batches, nchannels)
        stacked_data = torch.stack([torch.stack(recs[key], dim=0) for key in recs.keys()], dim=2)
        # split the stacked data into common shot gathers
        # splited_data: (1, nt, nreceivers in cogs, nchannels)
        splited_data = torch.split(stacked_data, reccounts, dim=1)
        y.data.extend(splited_data)
        
        # check if there is NaN in the output
        y.has_nan()

        return y

class WaveRNNJAX(WaveRNNBase):

    def __init__(self, cell, source_encoding=False, sharding=None):

        WaveRNNBase.__init__(self, cell, source_encoding, 'jax', sharding=sharding)

    def parameters(self):
        # for name in self.cell.geom.model_parameters:
        #     yield getattr(self.cell.geom, name)
        return tuple([getattr(self.cell.geom, name) for name in self.cell.geom.model_parameters])

    def set_parameters(self, parameters):
        for name, para in zip(self.cell.geom.model_parameters, parameters):
            setattr(self.cell.geom, name, para)

    def forward(self, x, omega=0.0, super_source=None, super_probes=None, parameters=None):

        # initialize the wavefields on target device
        # also initialize the self.nshots and self.hidden_state_shape
        self.initialize(super_source.x.shape[0])

        # Add source mask
        super_source.smask = jnp.zeros(self.hidden_state_shape)
        bidx = jnp.arange(self.nshots) if not self.source_encoding else jnp.zeros(self.nshots,dtype=jnp.int32)
        super_source.smask = super_source.smask.at[bidx, super_source.y, super_source.x].set(1)

        source_idx_at = []
        receiver_idx_at = []

        for source_type in self.cell.geom.source_type:
            source_idx_at.append(self.wavefield_names.index(source_type))

        for receiver_type in self.cell.geom.receiver_type:
            receiver_idx_at.append(self.wavefield_names.index(receiver_type))

        channels = len(self.cell.geom.receiver_type)
        nrecs = super_probes.x.size//self.nshots
        nt = self.cell.geom.nt

        rec = jnp.zeros((self.nshots, nt, nrecs, channels))
        super_source.source_encoding = self.source_encoding
        def step_fn(carry, it):
            
            wavefields, modelparas, others, rec = carry

            # Forward
            wavefields = self.cell(wavefields, modelparas, **others)

            # Apply source
            wavefields = list(wavefields)
            for sidx in source_idx_at:
                wavefields[sidx] = super_source(wavefields[sidx], x[..., it])
            wavefields = tuple(wavefields)

            # Measure probe(s)
            for channel, ridx in enumerate(receiver_idx_at):
                rec_this_step = jnp.array(jnp.split(super_probes(wavefields[ridx]), self.nshots))
                rec = rec.at[:, it, :, channel].set(rec_this_step)
            
            return (wavefields, modelparas, others, rec), None

        # wavefields, model_vars, **kwargs
        wavefields = tuple([getattr(self, name) for name in self.wavefield_names])
        model_vars = self.parameters() if parameters is None else parameters
        other_params = dict(is_last_frame=None, source=None)

        initial = (wavefields, model_vars, other_params, rec)
        (final), _ = lax.scan(step_fn, initial, jnp.arange(nt))
        rec = final[-1]

        return rec

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)