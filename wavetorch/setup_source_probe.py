import torch
from .source import WaveSource
from .probe import WaveIntensityProbe

def setup_src_coords(coords, Npml):
    # Coordinate are specified
    src_x, src_y = coords
    return WaveSource(src_y+Npml, src_x+Npml)

def setup_rec_coords(coords, Npml):
    rec_x, rec_y = coords
    y = [py+Npml for py in rec_y]
    x = [px+Npml for px in rec_x]

    return [WaveIntensityProbe(torch.Tensor(y).type(torch.long), torch.Tensor(x).type(torch.long))]