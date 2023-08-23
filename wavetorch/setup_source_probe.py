import torch
from .source import WaveSource
from .probe import WaveIntensityProbe

def setup_src_coords(coords, Npml, multiple=False):
    # Coordinate are specified
    src_x, src_y = coords
    Npmlx = Npml
    Npmly = 0 if multiple else Npml
    return WaveSource(src_y+Npmly, src_x+Npmlx)

def setup_rec_coords(coords, Npml, multiple=False):
    rec_x, rec_y = coords
    Npmlx = Npml
    Npmly = 0 if multiple else Npml
    y = [py+Npmly for py in rec_y]
    x = [px+Npmlx for px in rec_x]

    return [WaveIntensityProbe(torch.Tensor(y).type(torch.long), torch.Tensor(x).type(torch.long))]