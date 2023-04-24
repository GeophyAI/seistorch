import torch
import numpy as np
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

    def forward(self, x, output_fields=False):
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
        txx = torch.zeros(hidden_state_shape, device=device)
        tzz = torch.zeros(hidden_state_shape, device=device)
        txz = torch.zeros(hidden_state_shape, device=device)

        #x = x.to(device)
        vx_all = []
        vz_all = []

        # Because these will not change with time we should pull them out here to avoid unnecessary calculations on each
        # tme step, dramatically reducing the memory load from backpropagation
        
        vp = self.cell.geom.__getattr__('vp')
        vs = self.cell.geom.__getattr__('vs')
        rho = self.cell.geom.__getattr__('rho')
        save_interval = self.cell.geom.save_interval
        # Loop through time
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):

            # Propagate the fields
            vx, vz, txx, tzz, txz = self.cell(vx, vz, txx, tzz, txz, [vp, vs, rho], i, save_interval)

            # Inject source(s)
            for source in self.sources:
                # Add source to each wavefield
                for source_type in self.cell.geom.source_type:
                    var = eval(source_type)
                    var.data = source(var, xi.squeeze(-1))

            if len(self.probes) > 0 and not output_fields:
                # Measure probe(s)
                probe_values_vx = []
                probe_values_vz = []
                for probe in self.probes:
                    vx_all.append(probe(vx))
                    vz_all.append(probe(vz))
                    # probe_values_vx.append(probe(vx))
                    # probe_values_vz.append(probe(vz))
                # vx_all.append(torch.stack(probe_values_vx, dim=-1))
                # vz_all.append(torch.stack(probe_values_vz, dim=-1))

        # Combine outputs into a single tensor
        y_vx = torch.stack(vx_all, dim=1).permute(1, 2, 0)
        y_vz = torch.stack(vz_all, dim=1).permute(1, 2, 0)
        y = torch.concatenate([y_vx, y_vz], dim = 2)
        return y