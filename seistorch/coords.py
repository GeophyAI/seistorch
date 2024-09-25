
import torch
import numpy as np
from jax import numpy as jnp

from seistorch.setup import setup_src_coords, setup_rec_coords
from .source import WaveSourceJax, WaveSourceTorch
from .probe import WaveProbeJax, WaveProbeTorch

def offset_with_boundary(src, rec, cfg):

    """Padding the source and receiver locations with boundary.

    Args:
        src (Array): The source coordinates (nshots, ndim).
        rec (Array): The receiver coordinates (nshots, ndim, nreceivers).
        cfg (Array): The configure file.

    Returns:
        src (Array): The source coordinates with respect to boundary.
        rec (Array): The receiver coordinates with respect to boundary.
    """

    bwidth = cfg['geom']['boundary']['width']
    multiple = cfg['geom']['multiple']

    ndims = src.shape[-1]

    # with top boundary
    src += bwidth
    rec += bwidth

    if multiple: # no top boundary
        src[:,-1] -= bwidth
        rec[:,-1, :] -= bwidth

    return src, rec


def setup_acquisition(src_list, rec_list, cfg, *args, **kwargs):

    use_jax = (cfg['backend'] == 'jax')
    use_torch = (cfg['backend'] == 'torch')

    bwidth = cfg['geom']['boundary']['width']
    multiple = cfg['geom']['multiple']

    sources, receivers = [], []

    for i in range(len(src_list)):
        src = setup_src_coords(src_list[i], bwidth, multiple, use_jax)
        rec = setup_rec_coords(rec_list[i], bwidth, multiple, use_jax)
        sources.append(src)
        receivers.extend(rec)

    return sources, receivers

def merge_sources_with_same_keys(sources, use_jax=False):
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
        if use_jax:
            batchindices.append(bidx*jnp.ones(1, dtype=jnp.int32))
        else:
            batchindices.append(bidx*torch.ones(1, dtype=torch.int64))

    return batchindices, super_source

def merge_receivers_with_same_keys(receivers, use_jax=False):
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
        if use_jax:
            batchindices.append(bidx*jnp.ones(_reccounts, dtype=jnp.int32))
        else:
            batchindices.append(bidx*torch.ones(_reccounts, dtype=torch.int64))
        
    # stack the coords
    for key in super_probes.keys():
        if use_jax:
            super_probes[key] = jnp.concatenate(super_probes[key], axis=0)
        else:
            super_probes[key] = torch.concatenate(super_probes[key], dim=0)

    if use_jax:
        reccounts = jnp.array(reccounts)
        batchindices = jnp.concatenate(batchindices)
    else:
        # reccounts = torch.tensor(reccounts, dtype=torch.int64)
        batchindices = torch.concatenate(batchindices)

    return reccounts, batchindices, super_probes

def single2batch2(src, rec, cfg, dev):

    use_jax = (cfg['backend'] == 'jax')
    use_torch = (cfg['backend'] == 'torch')

    nshots = src.shape[0]
    ndim   = src.shape[1]
    sources, receivers = [], []
    # Coordinate are specified
    keys = ['x', 'y', 'z']

    ws = WaveSourceJax if use_jax else WaveSourceTorch
    wp = WaveProbeJax if use_jax else WaveProbeTorch

    # Map to WaveSource and WaveProbe instances
    sources = list(map(lambda shot: ws(**{key: src[shot][i] for i, key in enumerate(keys[:ndim])}), range(nshots)))
    receivers = list(map(lambda shot: wp(**{key: rec[shot][i] for i, key in enumerate(keys[:ndim])}), range(nshots)))

    # Zip to batch
    bidx_source, sourcekeys = merge_sources_with_same_keys(sources, use_jax)
    reccounts, bidx_receivers, reckeys = merge_receivers_with_same_keys(receivers, use_jax)

    # Construct the batched source and batched probes instances
    batched_source = ws(bidx_source, **sourcekeys)
    batched_probes = wp(bidx_receivers, **reckeys)

    batched_probes.reccounts = reccounts

    return batched_source, batched_probes

def single2batch(src, rec, cfg, dev):
    """This function is used to convert the single source and receiver to batched source and receiver.

    Args:
        src (list): Python list of source coordinates (in grid).
        rec (list): Python list of receiver coordinates (in grid).
        cfg (dict): The configure file.
        dev (str): The device to use.

    Returns:
        super_source: The super source (<WaveSourceTorch> or <WaveSourceJax>).
        super_probes: The super probes.
    """

    use_jax = (cfg['backend'] == 'jax')
    use_torch = (cfg['backend'] == 'torch')
    # detect the type of rec

    if use_torch:
        rec = rec.permute(2, 0, 1).cpu().numpy().tolist()
        src = torch.stack(src).cpu().numpy().T.tolist()
    
    if use_jax:
        # rec = rec.transpose(1, 2, 0)
        src = jnp.stack(src).T#.tolist()

    # For setup aquisition, the shape of rec must be (batchsize, ndim, nrecs)
    # Padding the source and receiver locations with boundary
    padded_src, padded_rec = setup_acquisition(src, rec, cfg)

    if use_torch:
        if isinstance(padded_src, list):
            padded_src = torch.nn.ModuleList(padded_src)
        else:
            padded_src = torch.nn.ModuleList([padded_src])

        if isinstance(padded_rec, list):
            padded_rec = torch.nn.ModuleList(padded_rec)
        else:
            padded_rec = torch.nn.ModuleList([padded_rec])

    bidx_source, sourcekeys = merge_sources_with_same_keys(padded_src, use_jax)
    reccounts, bidx_receivers, reckeys = merge_receivers_with_same_keys(padded_rec, use_jax)

    # Get the source and receiver classes
    wavesource = WaveSourceJax if use_jax else WaveSourceTorch
    waveprobe = WaveProbeJax if use_jax else WaveProbeTorch

    # Construct the batched source and batched probes
    super_source = wavesource(bidx_source, **sourcekeys)
    super_probes = waveprobe(bidx_receivers, **reckeys)

    super_probes.reccounts = reccounts

    return super_source, super_probes