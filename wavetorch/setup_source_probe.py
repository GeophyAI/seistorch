import torch
from .source import WaveSource
from .probe import WaveIntensityProbe

def setup_src_coords(src_x, src_y, Nx, Ny, Npml):
    if (src_x is not None) and (src_y is not None):
        # Coordinate are specified
        return [WaveSource(src_x, src_y)]
    else:
        # Center at left
        return [WaveSource(Npml + 20, int(Ny / 2))]


def setup_src_coords_customer(src_x, src_y, Nx, Ny, Npml):
    if (src_x is not None) and (src_y is not None):
        # Coordinate are specified
        return WaveSource(src_y+Npml, src_x+Npml)
    else:
        # Center at left
        return [WaveSource(Npml + 20, int(Ny / 2))]

def setup_probe_coords(N_classes, px, py, pd, Nx, Ny, Npml):
    if (py is not None) and (px is not None):
        # All probe coordinate are specified
        assert len(px) == len(py), "Length of px and py must match"

        return [WaveIntensityProbe(px[j], py[j]) for j in range(0, len(px))]

    if (py is None) and (pd is not None):
        # Center the probe array in y
        span = (N_classes - 1) * pd
        y0 = int((Ny - span) / 2)
        assert y0 > Npml, "Bottom element of array is inside the PML"
        y = [y0 + i * pd for i in range(N_classes)]

        if px is not None:
            assert len(px) == 1, "If py is not specified then px must be of length 1"
            x = [px[0] for i in range(N_classes)]
        else:
            x = [Nx - Npml - 20 for i in range(N_classes)]
        #return [WaveIntensityProbe(x[j], y[j]) for j in range(0, len(x))]

    raise ValueError("px = {}, py = {}, pd = {} is an invalid probe configuration".format(pd))
    

def setup_probes_moving(probe_x, probe_y, Nx, Ny, Npml):
    return [WaveIntensityProbe(torch.Tensor([probe_y+Npml]).type(torch.long), torch.Tensor([probe_x+Npml]).type(torch.long))]

def setup_probe_coords_customer(cfg):
    Nreceivers = cfg['geom']['Nreceivers']
    init_px = cfg['geom']['ipx']
    py = cfg['geom']['py']
    pd = cfg['geom']['pd']
    Nx = cfg['geom']['Nx']
    Ny = cfg['geom']['Ny']
    Npml = cfg['geom']['pml']['N']

    if (init_px is not None) and (py is not None):
    
        if pd is not None:
            pd = pd
        else:
            pd = (Nx-2*Npml-init_px)//Nreceivers

        y = [py+Npml for _ in range(Nreceivers)]
        assert (y[0] >= Npml and y[0] <= Ny-Npml), "Receivers are inside the PML"

        x = [init_px+Npml+pd*i for i in range(Nreceivers)]
        assert (x[0] >= Npml and x[-1] <= Nx-Npml), "Receivers are inside the PML"

        #return [WaveIntensityProbe(y[j], x[j]) for j in range(0, len(x))]
        return [WaveIntensityProbe(torch.Tensor(y).type(torch.long), torch.Tensor(x).type(torch.long))]
    raise ValueError("px = {}, py = {}, pd = {} is an invalid probe configuration".format(pd))


def get_sources_coordinate_list(cfg):
    ipx = cfg['geom']['isx']
    src_y = cfg['geom']['src_y']
    offset = cfg['geom']['src_d']
    Nshots = cfg['geom']['Nshots']
    """get the x and y coordinate list for forward mode.
       width of pml will be added in func<setup_src_coords_customer>
       and func<setup_probe_coords_customer>
    """
    source_y_list = [src_y for _ in range(Nshots)]
    source_x_list = [ipx + offset*shot for shot in range(Nshots)]
    return source_x_list, source_y_list
