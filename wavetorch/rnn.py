import torch
from .wavefield import Wavefield
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

            # Propagate the fields
            wavefield = [self.__getattribute__(name) for name in wavefield_names]
            wavefield = self.cell(wavefield, model_paras, t=i, it=save_interval, omega=omega)
            
            # Set the data to vars
            for name, data in zip(wavefield_names, wavefield):
                # self.__getattribute__(name).copy_(data)
                self.__setattr__(name, data)

            # Add source
            for source, _s in zip(self.sources, xi.chunk(xi.size(0), dim=0)):
                for source_type in self.cell.geom.source_type:
                    #self.h1[:,source.x, source.y] += _s.squeeze(-1)
                    self.__setattr__(source_type, source(self.__getattribute__(source_type), _s.squeeze(-1)))

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
    

    # """TBPTT"""
    # def forward(self, x, omega=10.0, obs_y=None, criterion=None, opt=None):
    #     device = self.cell.geom.device
    #     batch_size = x.shape[0]
    #     hidden_state_shape = (1,) + self.cell.geom.domain_shape

    #     wavefield_names = Wavefield(self.cell.geom.equation).wavefields

    #     for name in wavefield_names:
    #         self.__setattr__(name, torch.zeros(hidden_state_shape, device=device))

    #     for name in self.cell.geom.model_parameters:
    #         self.__setattr__(name, self.cell.geom.__getattr__(name))

    #     x = x.to(device)

    #     sequence_length = x.size(1)
    #     truncated_length = 500
    #     p_all = []
    #     initial_hidden_states = {name: self.__getattribute__(name).clone() for name in wavefield_names}
        
    #     grad = torch.zeros_like(self.cell.geom.vp)

    #     for start_idx in range(0, sequence_length, truncated_length):
    #         # opt.zero_grad()
    #         end_idx = min(start_idx + truncated_length, sequence_length)
    #         p_all_truncated, hidden_states = self.forward_truncated_sequence(x, omega, start_idx, end_idx, initial_hidden_states)
    #         p_all.extend(p_all_truncated)
    #         initial_hidden_states = {k: v for k, v in hidden_states.items()}
    #         #initial_hidden_states = hidden_states.copy()
    #         y_truncated = torch.cat([torch.stack(y, dim=1).permute(1, 2, 0) for y in p_all_truncated], dim=1)
    #         if obs_y is not None: 
    #             y_obs_truncated = obs_y[start_idx:end_idx].clone()
    #             assert y_obs_truncated.shape == y_truncated.shape, \
    #                 f"y_obs_truncated and y_truncated must have the same shape, \
    #                     but got {y_obs_truncated.shape} and {y_truncated.shape}"

    #             # Calculate the loss for the current truncated sequence
    #             loss = criterion(y_truncated, y_obs_truncated)
    #             # Backpropagate the gradients and update the optimizer
    #             loss.backward(retain_graph=True)
    #             grad+=self.cell.geom.vp.grad


    #     #print(grad.max(), grad.min())
    #     np.save("/mnt/data/wangsw/inversion/marmousi_20m/results/test_autodiff/selfgrad.npy", self.cell.geom.vp.grad.cpu().detach().numpy())
    #     np.save("/mnt/data/wangsw/inversion/marmousi_20m/results/test_autodiff/grad.npy", grad.cpu().detach().numpy())

    #     y = torch.cat([torch.stack(y, dim=1).permute(1, 2, 0) for y in p_all], dim=0)  # Changed from `concat` to `cat` and updated the concatenation dimension to 1
    #     has_nan = torch.isnan(x).any()
    #     assert not has_nan, "Warning!!Data has nan!!"
    #     return y

    # def forward_truncated_sequence(self, x, omega, start_idx, end_idx, initial_hidden_states):
    #     wavefield_names = Wavefield(self.cell.geom.equation).wavefields
    #     length_record = len(self.cell.geom.receiver_type)
    #     p_all = [[] for i in range(length_record)]
    #     model_paras = [self.__getattr__(name) for name in self.cell.geom.model_parameters]
    #     save_interval = self.cell.geom.save_interval

    #     # Set initial hidden states
    #     for name, hidden_state in initial_hidden_states.items():
    #         self.__setattr__(name, hidden_state)

    #     for i, xi in enumerate(x.chunk(x.size(1), dim=1)[start_idx:end_idx]):
    #         wavefield = [self.__getattribute__(name) for name in wavefield_names]
    #         wavefield = self.cell(wavefield, model_paras, t=i, it=save_interval, omega=omega)

    #         for name, data in zip(wavefield_names, wavefield):
    #             self.__setattr__(name, data)

    #         for source, _s in zip(self.sources, xi.chunk(xi.size(0), dim=0)):
    #             for source_type in self.cell.geom.source_type:
    #                 self.__setattr__(source_type, source(self.__getattribute__(source_type), _s.squeeze(-1)))

    #         if len(self.probes) > 0:
    #             for probe in self.probes:
    #                 for receiver, p_all_sub in zip(self.cell.geom.receiver_type, p_all):
    #                     p_all_sub.append(probe(self.__getattribute__(receiver)))

    #     hidden_states = {name: self.__getattribute__(name).clone() for name in wavefield_names}
    #     return p_all, hidden_states


