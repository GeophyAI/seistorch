import torch
import numpy as np
import matplotlib.pyplot as plt
from .utils import diff_using_roll
class WaveRNN(torch.nn.Module):
    def __init__(self, cell, sources=None, probes=[]):

        super().__init__()

        self.cell = cell

        if type(probes) is list:
            self.probes = torch.nn.ModuleList(probes)
        else:
            self.probes = torch.nn.ModuleList([probes])

    def reset_sources(self, sources):
        if type(sources) is list:
            self.sources = torch.nn.ModuleList(sources)
        else:
            self.sources = torch.nn.ModuleList([sources])

    def reset_probes(self, probes):
        if type(probes) is list:
            self.probes = torch.nn.ModuleList(probes)
        else:
            self.probes = torch.nn.ModuleList([probes])

    def forward(self, x, omega=10.0):
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
        # First dim is batch
        batch_size = x.shape[0]
        # Init hidden states
        hidden_state_shape = (batch_size,) + self.cell.geom.domain_shape

        vx = torch.zeros(hidden_state_shape, device=device)
        vz = torch.zeros(hidden_state_shape, device=device)
        p = torch.zeros(hidden_state_shape, device=device)
        r = torch.zeros(hidden_state_shape, device=device)

        p_all = []

        # Because these will not change with time we should pull them out here to avoid unnecessary calculations on each
        # tme step, dramatically reducing the memory load from backpropagation
        
        vp = self.cell.geom.__getattr__('vp')
        rho = self.cell.geom.__getattr__('rho')
        Q = self.cell.geom.__getattr__('Q')

        save_interval = self.cell.geom.save_interval
        # Loop through time

        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):

            # Propagate the fields
            vx, vz, p, r = self.cell(vx, vz, p, r, [vp, rho, Q], i, save_interval, omega)

            # Inject source(s)
            for source in self.sources:
               # Add source to each wavefield
               p = source(p, xi.squeeze(-1))

            if len(self.probes) > 0:
                # Measure probe(s)
                for probe in self.probes:
                    p_all.append(probe(p))

        # Combine outputs into a single tensor
        y = torch.stack(p_all, dim=1).permute(1, 2, 0)
        has_nan = torch.isnan(x).any()
        assert not has_nan, "Warning!!Data has nan!!"
        return y