import torch
from .wavefield import Wavefield

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
        hidden_state_shape = (1,) + self.cell.geom.domain_shape

        wavefield_names = Wavefield(self.cell.geom.equation).wavefields

        # Set wavefields
        for name in wavefield_names:
            self.__setattr__(name, torch.zeros(hidden_state_shape, device=device))

        length_record = len(self.cell.geom.receiver_type)
        p_all = [[] for i in range(length_record)]

        # Set model parameters
        for name in self.cell.geom.model_parameters:
            self.__setattr__(name, self.cell.geom.__getattr__(name))
        
        # Pack parameters
        model_paras = [self.__getattr__(name) for name in self.cell.geom.model_parameters]
        
        # Short cut of the save intervel
        save_interval = self.cell.geom.save_interval

        #src_index = [[0, source._x, source._y] for source in self.sources]
            
        # Loop through time
        x = x.to(device)
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            #print(i)
            # Propagate the fields
            wavefield = [self.__getattribute__(name) for name in wavefield_names]
            wavefield = self.cell(wavefield, model_paras, t=i, it=save_interval, omega=omega)
            # Set the data to vars
            for name, data in zip(wavefield_names, wavefield):
                self.__setattr__(name, data)

            # Inject source(s)
            #print(xi.chunk(xi.size(0), dim=0))

            for source, _s in zip(self.sources, xi.chunk(xi.size(0), dim=0)):
                for source_type in self.cell.geom.source_type:
                    self.__setattr__(source_type, source(self.__getattribute__(source_type), _s.squeeze(-1)))

            
            # for source, _s in zip(self.sources, xi.chunk(xi.size(0), dim=0)):
                #self.h1[0, source.x, source.y] += _s.squeeze()
                # for source_type in self.cell.geom.source_type:
                    # Add source to each wavefield
                    # source_var = self.__getattribute__(source_type)
                    # source_var[0, source.x, source.y] += _s.squeeze()
                    #self.__setattr__(source_type, source(self.__getattribute__(source_type), _s.squeeze(-1)))
                
            if len(self.probes) > 0:
                # Measure probe(s)
                for probe in self.probes:
                    for receiver, p_all_sub in zip(self.cell.geom.receiver_type, p_all):
                        p_all_sub.append(probe(self.__getattribute__(receiver)))

        # Combine outputs into a single tensor
        y = torch.concat([torch.stack(y, dim=1).permute(1, 2, 0) for y in p_all], dim = 2)
        has_nan = torch.isnan(x).any()
        assert not has_nan, "Warning!!Data has nan!!"
        return y