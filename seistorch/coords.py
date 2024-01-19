
import torch

from seistorch import WaveSource, WaveProbe, WaveIntensityProbe


def setup_rec_coords(coords, Npml, multiple=False):
    """Setup receiver coordinates.

    Args:
        coords (list): A list of coordinates.
        Npml (int): The number of PML layers.
        multiple (bool, optional): Whether use top PML or not. Defaults to False.

    Returns:
        WaveProbe: A torch.nn.Module receiver object.
    """

    # Coordinate are specified
    keys = ['x', 'y', 'z']
    kwargs = dict()

    # Without multiple
    for key, value in zip(keys, coords):
        kwargs[key] = [v + Npml if v is not None else None for v in value]

    # 2D case with multiple
    if 'z' not in kwargs.keys() and multiple:
        kwargs['y'] = [v - Npml if v is not None else None for v in kwargs['y']]

    # 3D case with multiple
    if 'z' in kwargs.keys() and multiple:
        raise NotImplementedError("Multiples in 3D case is not implemented yet.")
        #kwargs['z'] = [v-Npml for v in kwargs['z']]

    return [WaveIntensityProbe(**kwargs)]

def setup_src_coords(coords, Npml, multiple=False):
    """Setup source coordinates.

    Args:
        coords (list): A list of coordinates.
        Npml (int): The number of PML layers.
        multiple (bool, optional): Whether use top PML or not. Defaults to False.

    Returns:
        WaveSource: A torch.nn.Module source object.
    """
    # Coordinate are specified
    keys = ['x', 'y', 'z']
    kwargs = dict()
    # Padding the source location with PML
    for key, value in zip(keys, coords):
        if isinstance(value, (int, float)):
            kwargs[key] = value+Npml
        else:
            kwargs[key] = value # value = None

    # 2D case with multiple
    if 'z' not in kwargs.keys() and multiple and bool(kwargs['y']):
        kwargs['y'] -= Npml

    # 3D case with multiple
    if 'z' in kwargs.keys() and multiple:
        raise NotImplementedError("Multiples in 3D case is not implemented yet.")
        # kwargs['z'] -= Npml

    return WaveSource(**kwargs)

def setup_acquisition(src_list, rec_list, cfg, *args, **kwargs):

    sources, receivers = [], []

    for i in range(len(src_list)):
        src = setup_src_coords(src_list[i], cfg['geom']['pml']['N'], cfg['geom']['multiple'])
        rec = setup_rec_coords(rec_list[i], cfg['geom']['pml']['N'], cfg['geom']['multiple'])
        sources.append(src)
        receivers.extend(rec)

    return sources, receivers

def merge_sources_with_same_keys(sources):
    """Merge all source coords into a super shot.
    """
    super_source = dict()
    batchindices = []

    for bidx, source in enumerate(sources):
        coords = source.coords()
        for key in coords.keys():
            if key not in super_source.keys():
                super_source[key] = []
            super_source[key].append(coords[key])
        batchindices.append(bidx*torch.ones(1, dtype=torch.int64))

    return batchindices, super_source

def merge_receivers_with_same_keys(receivers):
    """Merge all source coords into a super shot.
    """
    super_probes = dict()
    batchindices = []
    reccounts = []
    for bidx, probe in enumerate(receivers):
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

def single2batch(src, rec, cfg, dev):

    rec = [torch.stack(item) for item in rec]
    rec = torch.stack(rec).permute(2, 0, 1).cpu().numpy().tolist()

    src = torch.stack(src).cpu().numpy().T.tolist()

    padded_src, padded_rec = setup_acquisition(src, rec, cfg)

    if isinstance(padded_src, list):
        padded_src = torch.nn.ModuleList(padded_src)
    else:
        padded_src = torch.nn.ModuleList([padded_src])

    if isinstance(padded_rec, list):
        padded_rec = torch.nn.ModuleList(padded_rec)
    else:
        padded_rec = torch.nn.ModuleList([padded_rec])

    # Get the super source and super probes
    bidx_source, sourcekeys = merge_sources_with_same_keys(padded_src)
    super_source = WaveSource(bidx_source, **sourcekeys).to(dev) 

    reccounts, bidx_receivers, reckeys = merge_receivers_with_same_keys(padded_rec)
    super_probes = WaveProbe(bidx_receivers, **reckeys).to(dev)
    super_probes.reccounts = reccounts
    
    return super_source, super_probes