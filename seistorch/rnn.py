import numpy as np
import torch

from .eqconfigure import Wavefield
from .source import WaveSource
from .probe import WaveProbe
from .cell import WaveCell
from .utils import merge_dicts_with_same_keys

class WaveRNN(torch.nn.Module):
    def __init__(self, cell, source_encoding=False):

        super().__init__()

        self.cell = cell
        #  Check the availability of the type of sources and probes.
        self.check()
        self.source_encoding = source_encoding # short cut

    def check(self,):
        
        wavefield_names = Wavefield(self.cell.geom.equation).wavefields

        # Check source type:
        for source_type in self.cell.geom.source_type:
            assert source_type in wavefield_names, \
                f"Valid source type are {wavefield_names}, but got '{source_type}'. Please check your configure file."

        # Check receiver type:
        for recev_type in self.cell.geom.receiver_type:
            assert recev_type in wavefield_names, \
                f"Valid receiver type are {wavefield_names}, but got '{recev_type}'. Please check your configure file."

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
        if type(sources) is list:
            self.sources = torch.nn.ModuleList(sources)
        else:
            self.sources = torch.nn.ModuleList([sources])

    def reset_probes(self, probes):
        if type(probes) is list:
            self.probes = torch.nn.ModuleList(probes)
        else:
            self.probes = torch.nn.ModuleList([probes])
        

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
        
        length_record = len(self.cell.geom.receiver_type)
        p_all = [[] for i in range(length_record)]

        # Set model parameters
        for name in self.cell.geom.model_parameters:
            setattr(self, name, getattr(self.cell.geom, name))

        # Pack parameters
        model_paras = [getattr(self, name) for name in self.cell.geom.model_parameters]

        # Loop through time
        x = x.to(device)

        super_source = WaveSource(**self.merge_sources_with_same_keys()).to(device)
        super_probes = WaveProbe(**self.merge_receivers_with_same_keys()).to(device)
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

            # Set the data to vars
            for name, data in zip(wavefield_names, wavefield):
                setattr(self, name, data)

            # Add source
            for source_type in self.cell.geom.source_type:
                setattr(self, source_type, super_source(getattr(self, source_type), xi.view(xi.size(1), -1)))

            # Measure probe(s)
            #for probe in self.probes:
            for receiver, p_all_sub in zip(self.cell.geom.receiver_type, p_all):
                #p_all_sub.append(probe(getattr(self, receiver)))
                p_all_sub.append(super_probes(getattr(self, receiver)))

        # Combine outputs into a single tensor
        permute_axis = (0, 1, 3, 2) if torch.stack(p_all[0], dim=1).dim() == 4 else (1, 2, 3, 0)

        y = torch.concat([torch.stack(y, dim=1).permute(*permute_axis) for y in p_all], dim = 3)
        
        has_nan = torch.isnan(y).any()
        assert not has_nan, "Warning!!Data has nan!!"
        return y