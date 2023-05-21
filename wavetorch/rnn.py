import numpy as np
import torch

from .wavefield import Wavefield
from .source import WaveSource
from .utils import diff_using_roll

def _time_step(*args):

    vp, rho = args[0:2]
    vx, vz, p = args[2:5]
    dt, h, b = args[5:8]
    # 更新速度场
    c = 0.5*dt*b

    p_x = diff_using_roll(p, 2, False)
    p_z = diff_using_roll(p, 1, False)

    y_vx = dt * diff_using_roll(p, 2, False) / h - dt  * vx
    y_vz = dt * diff_using_roll(p, 1, False) / h - dt  * vz


    # y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*p_x+(1-c)*vx)
    # y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*p_z+(1-c)*vz)

    # x -- 2
    # z -- 1
    # vx_x = diff_using_roll(y_vx, 2)
    # vz_z = diff_using_roll(y_vz, 1)
    # 更新力场
    y_p = dt * vp**2 * (
         diff_using_roll(y_vx, dim=2) / h +
         diff_using_roll(y_vz, dim=1) / h
    ) - dt * p

    # y_p = (1+c)**-1*(vp**2*dt*rho.pow(-1)*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)

    return y_vx, y_vz, y_p


class WaveRNN(torch.nn.Module):
    def __init__(self, cell, sources=None, probes=[]):

        super().__init__()

        self.cell = cell

        self.check()

        # if type(probes) is list:
        #     self.probes = torch.nn.ModuleList(probes)
        # else:
        #     self.probes = torch.nn.ModuleList([probes])


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

        # Loop through time
        x = x.to(device)
        super_source = WaveSource([s.x for s in self.sources], 
                                  [s.y for s in self.sources]).to(device)

        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):

            # if i %1==0:
            #     np.save(f"/mnt/data/wangsw/inversion/marmousi_10m/inv_rho/l2/forward/forward{i:04d}.npy", 
            #             self.p.cpu().detach().numpy())
                
            # Propagate the fields
            wavefield = [self.__getattribute__(name) for name in wavefield_names]

            wavefield = self.cell(wavefield, 
                                  model_paras, 
                                  is_last_frame=(i==x.size(1)-1), 
                                  omega=omega, 
                                  source=[self.cell.geom.source_type, super_source, xi.view(xi.size(1), -1)])


            # Set the data to vars
            for name, data in zip(wavefield_names, wavefield):
                self.__setattr__(name, data)

            # # Add source
            for source_type in self.cell.geom.source_type:
                self.__setattr__(source_type, super_source(self.__getattribute__(source_type), xi.view(xi.size(1), -1)))

            # Measure probe(s)
            for probe in self.probes:
                for receiver, p_all_sub in zip(self.cell.geom.receiver_type, p_all):
                    p_all_sub.append(probe(self.__getattribute__(receiver)))

        # Combine outputs into a single tensor
        y = torch.concat([torch.stack(y, dim=1).permute(1, 2, 0) for y in p_all], dim = 2)
        has_nan = torch.isnan(y).any()
        assert not has_nan, "Warning!!Data has nan!!"
        return y