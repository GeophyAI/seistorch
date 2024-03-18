import segyio
import numpy as np
import matplotlib.pyplot as plt

def read_dt(path):
    with segyio.open(path, ignore_geometry=True) as f:
        dt = segyio.dt(f) / 1000
    return dt

def get_resort_dict(path):
    dmap = {}
    with segyio.open(path, ignore_geometry=True) as f:
        # Add the data with same source coordinate in to a same list
        f.mmap()
        # Go trough all the traces
        for i,h in enumerate(f.header):
            # Get the source coordinate
            sourceX = h[segyio.TraceField.SourceX]
            sourceY = h[segyio.TraceField.SourceY]
            sourceZ = h[segyio.TraceField.SourceDepth]
            # Get the receiver coordinate
            if (sourceX, sourceY, sourceZ) in dmap:
                dmap[(sourceX, sourceY, sourceZ)].append(i)
            else:
                dmap[(sourceX, sourceY, sourceZ)] = []
                # dmap[(sourceX, sourceY, sourceZ)]['map_idx'] = []
                # dmap[(sourceX, sourceY, sourceZ)]['rec']
    return dmap

def extract(path, dmap):
    nshots = len(dmap)
    d = np.empty(nshots, dtype=np.ndarray)
    with segyio.open(path, ignore_geometry=True) as f:
        f.mmap()
        for i, (k, v) in enumerate(dmap.items()):
            temp = []
            for j in v:
                temp.append(f.trace.raw[j])
            d[i] = np.array(temp).T
    return d

filename = '/home/wangsw/wangsw/model/Model94_shots.segy'
dmap = get_resort_dict(filename)
d = extract(filename, dmap)

shot_no = 50
vmin, vmax = np.percentile(d[shot_no], [1, 99])
plt.figure(figsize=(5, 4))
plt.imshow(d[shot_no][:400], cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
plt.colorbar()
plt.title('Shot No.{}'.format(shot_no))
plt.show()

dt = read_dt(filename)
print('dt = {} ms'.format(dt))
