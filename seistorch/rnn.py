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
        batchindices = []

        for bidx, source in enumerate(self.sources):
            coords = source.coords()
            for key in coords.keys():
                if key not in super_source.keys():
                    super_source[key] = []
                super_source[key].append(coords[key])
            batchindices.append(bidx*torch.ones(1, dtype=torch.int64))

        return batchindices, super_source

    def merge_receivers_with_same_keys(self,):
        """Merge all source coords into a super shot.
        """
        super_probes = dict()
        batchindices = []
        reccounts = []
        for bidx, probe in enumerate(self.probes):
            coords = probe.coords()
            for key in coords.keys():
                if key not in super_probes.keys():
                    super_probes[key] = []
                super_probes[key].append(coords[key])
            # how many receivers in this group
            _reccounts = len(coords[key])
            # add reccounts and batchindices
            reccounts.append(_reccounts)
            batchindices.append(bidx*torch.ones(_reccounts, dtype=torch.int64))
            
        # stack the coords
        for key in super_probes.keys():
            super_probes[key] = torch.concatenate(super_probes[key], dim=0)

        return reccounts, torch.concatenate(batchindices), super_probes

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
            recs[key] = []

        # p_all = [[] for i in range(nchannels)]

        # Set model parameters
        for name in self.cell.geom.model_parameters:
            setattr(self, name, getattr(self.cell.geom, name))

        # Pack parameters
        model_paras = [getattr(self, name) for name in self.cell.geom.model_parameters]

        # Loop through time
        x = x.to(device)
        # Get the super source and super probes
        bidx_source, sourcekeys = self.merge_sources_with_same_keys()
        super_source = WaveSource(bidx_source, **sourcekeys).to(device)
        
        reccounts, bidx_receivers, reckeys = self.merge_receivers_with_same_keys()
        super_probes = WaveProbe(bidx_receivers, **reckeys).to(device)
        
        super_source.source_encoding = self.source_encoding

        time_offset = 2 if self.cell.geom.equation == "acoustic" else 0
        
        batched_records = []
        
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
                np.save(f"./wf_pml/wf{i:04d}.npy", wavefield[0].cpu().numpy())

            # Set the data to vars
            for name, data in zip(wavefield_names, wavefield):
                setattr(self, name, data)

            # Add source
            for source_type in self.cell.geom.source_type:
                setattr(self, source_type, super_source(getattr(self, source_type), xi.view(xi.size(1), -1)))

            # Measure probe(s): new implementation
            for key in self.cell.geom.receiver_type:
                recs[key].append(super_probes(getattr(self, key)))

            
        # stacked_data: (nbatches, nt, nreceivers all batches, nchannels)
        stacked_data = torch.stack([torch.stack(recs[key], dim=0) for key in recs.keys()], dim=2)
        # split the stacked data into common shot gathers
        # splited_data: (1, nt, nreceivers in cogs, nchannels)
        splited_data = torch.split(stacked_data, reccounts, dim=1)
        y.data.extend(splited_data)
        
        # check if there is NaN in the output
        y.has_nan()

        return y