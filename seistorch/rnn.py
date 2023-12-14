import numpy as np
import torch

from .eqconfigure import Wavefield
from .source import WaveSource
from .probe import WaveProbe
from .cell import WaveCell
from .setup import setup_acquisition
from .type import TensorList

class WaveRNN(torch.nn.Module):
    def __init__(self, cell, source_encoding=False):

        super().__init__()

        self.cell = cell
        #  Check the availability of the type of sources and probes.
        self.source_encoding = source_encoding # short cut

    def merge_sources_with_same_keys(self,):
        """Merge all source coords into a super shot.
        """
        super_source = dict()
        for source in self.sources:
            coords = source.coords()
            for key in coords.keys():
                if key not in super_source.keys():
                    super_source[key] = []
                super_source[key].append(coords[key])
        return super_source

    def merge_receivers_with_same_keys(self,):
        """Merge all source coords into a super shot.
        """
        super_probes = dict()
        for probe in self.probes:
            coords = probe.coords()
            for key in coords.keys():
                if key not in super_probes.keys():
                    super_probes[key] = []
                super_probes[key].append(coords[key])

        # stack the coords
        for key in super_probes.keys():
            super_probes[key] = torch.stack(super_probes[key], dim=0)

        return super_probes

    def reset_sources(self, sources):
        if isinstance(sources, list):
            self.sources = torch.nn.ModuleList(sources)
        else:
            self.sources = torch.nn.ModuleList([sources])

    def reset_probes(self, probes):
        if isinstance(probes, list):
            self.probes = torch.nn.ModuleList(probes)
        else:
            self.probes = torch.nn.ModuleList([probes])

    def reset_geom(self, shots, src_list, rec_list, cfg):

        sources, receivers = setup_acquisition(shots, src_list, rec_list, cfg)

        self.reset_sources(sources)
        self.reset_probes(receivers)
        
    """Original implementation"""
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

        # Init hidden states
        batchsize = 1 if self.source_encoding else len(self.sources)
        hidden_state_shape = (batchsize,) + self.cell.geom.domain_shape
        wavefield_names = Wavefield(self.cell.geom.equation).wavefields
        # Set wavefields
        for name in wavefield_names:
            setattr(self, name, torch.zeros(hidden_state_shape, device=device))

        # nchannels = len(self.cell.geom.receiver_type)

        recs = dict()
        y = TensorList()
        for key in self.cell.geom.receiver_type:
            recs[key] = {}
            for igroup in range(len(self.probes)):
                recs[key][f"group{igroup}"] = []

        # p_all = [[] for i in range(nchannels)]

        # Set model parameters
        for name in self.cell.geom.model_parameters:
            setattr(self, name, getattr(self.cell.geom, name))

        # Pack parameters
        model_paras = [getattr(self, name) for name in self.cell.geom.model_parameters]

        # Loop through time
        x = x.to(device)

        super_source = WaveSource(**self.merge_sources_with_same_keys()).to(device)
        #super_probes = WaveProbe(**self.merge_receivers_with_same_keys()).to(device)
        super_source.source_encoding = self.source_encoding

        time_offset = 2 if self.cell.geom.equation == "acoustic" else 0
        
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
        
            # Propagate the fields
            # wavefield = [self.__getattribute__(name) for name in wavefield_names]
            wavefield = [getattr(self, name) for name in wavefield_names]
            tmpi = min(i+time_offset, x.shape[1]-1)
            wavefield = self.cell(wavefield, 
                                  model_paras, 
                                  is_last_frame=(i==x.size(1)-1), 
                                  omega=omega, 
                                  source=[self.cell.geom.source_type, super_source, x[..., tmpi].view(xi.size(1), -1)])
            
            if False:
                np.save(f"./wavefield_acoustic/wf{i:04d}.npy", wavefield[0].cpu().numpy())

            # Set the data to vars
            for name, data in zip(wavefield_names, wavefield):
                setattr(self, name, data)

            # Add source
            for source_type in self.cell.geom.source_type:
                setattr(self, source_type, super_source(getattr(self, source_type), xi.view(xi.size(1), -1)))

            # Measure probe(s)
            for key in self.cell.geom.receiver_type:
                for igroup, recgroup in enumerate(self.probes):
                    recs[key][f"group{igroup}"].append(recgroup(getattr(self, key))[igroup])

        # Combine outputs into a single tensor
        for igroup in range(len(self.probes)):
            for key in self.cell.geom.receiver_type:
                recs[key][f"group{igroup}"] = torch.stack(recs[key][f"group{igroup}"], dim=0)
            y.append(torch.stack([recs[key][f"group{igroup}"] for key in recs.keys()], dim=2))
        
        y.has_nan()

        return y