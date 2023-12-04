from typing import Any
import segyio
import numpy as np
import matplotlib.pyplot as plt

class Coordinate:

    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self) -> Any:
        return self._x
    
    @property
    def y(self) -> Any:
        return self._y
    
    @property
    def z(self) -> Any:
        return self._z

    def __str__(self):
        return 'x: {}, y: {}, z: {}'.format(self.x, self.y, self.z)
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Coordinate):
            return self.x == __value.x and self.y == __value.y and self.z == __value.z
        else:
            return False
        
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

class Trace:

    def __init__(self, src: Coordinate, rec: Coordinate, idx_in_segy: int = None):
        self._src = src
        self._rec = rec
        self._idx_in_segy = idx_in_segy

    @property
    def src(self):
        return (self._src.x, self._src.y, self._src.z)
    
    @property
    def rec(self):
        return (self._rec.x, self._rec.y, self._rec.z)
    
    @property
    def idx_in_segy(self):
        return self._idx_in_segy
    
class CommonShotGather:

    def __init__(self, src: Coordinate, recs: list, data: np.ndarray):
        self._src = np.asarray(src)
        self._recs = np.asarray(recs)
        self._data = data

    @property
    def src(self):
        return self._src
    
    @property
    def recs(self):
        return self._recs
    
    @property
    def data(self):
        return self._data
    
class SegyReader:

    def __init__(self, path):
        self.path = path
        self.analysis()
        self.map2csg()

    def _open(self):
        self.f = segyio.open(self.path, ignore_geometry=True)
        self.f.mmap()

    def _close(self):
        self.f.close()

    @property
    def dt(self):
        """Get the sample interval in ms"""
        self._open()
        dt = segyio.dt(self.f) / 1000
        self._close()
        return dt

    @property
    def tracecount(self):
        """Get the number of traces in the segy file"""
        self._open()
        num_traces = self.f.tracecount
        self._close()
        return num_traces
    
    @property
    def shotcount(self):
        """Get the number of shots in the segy file"""
        return len(self.dmap)

    def analysis(self):
        """Get the information of each trace in the segy file"""
        self._open()
        self.trace_info = []
        for i,h in enumerate(self.f.header):
            # Get the source coordinate
            srcX = h[segyio.TraceField.SourceX]
            srcY = h[segyio.TraceField.SourceY]
            srcZ = h[segyio.TraceField.SourceDepth]
            # Get the receiver coordinate
            recX = h[segyio.TraceField.GroupX]
            recY = h[segyio.TraceField.GroupY]
            recZ = h[segyio.TraceField.GroupWaterDepth]
            trace = Trace(Coordinate(srcX, srcY, srcZ), Coordinate(recX, recY, recZ), i)
            self.trace_info.append(trace)
        self._close()

    def map2csg(self):
        """Map the traces to common shot gathers"""
        self.dmap = dict()
        for i,t in enumerate(self.trace_info):
            # Get the receiver coordinate
            if t.src in self.dmap:
                self.dmap[t.src]['idx_in_segy'].append(t.idx_in_segy)
                self.dmap[t.src]['recs'].append(t.rec)
            else:
                self.dmap[t.src] = dict()
                self.dmap[t.src]['idx_in_segy'] = []
                self.dmap[t.src]['recs'] = []
        return self.dmap
    
    def get_shot(self, shot_no):
        """Get the data of shot gather by shot_no"""
        if shot_no >= len(self.dmap):
            raise ValueError('shot_no should be less than {}'.format(len(self.dmap)))
        if shot_no < 0:
            shot_no = len(self.dmap) + shot_no
        self._open()
        key = list(self.dmap.keys())[shot_no]
        idx_in_segy = self.dmap[key]['idx_in_segy']
        data = []
        for i in idx_in_segy:
            data.append(self.f.trace.raw[i])
        self._close()
        data = np.array(data).T

        cog = CommonShotGather(key, self.dmap[key]['recs'], data)

        return cog


filename = '/home/wangsw/wangsw/model/1997_2.5D_shots.segy'

reader = SegyReader(filename)
tracecount = reader.tracecount
shotcount = reader.shotcount
cog = reader.get_shot(200)

fig, ax = plt.subplots(figsize=(5, 4))
vmin, vmax = np.percentile(cog.data, [1, 99])
ax.imshow(cog.data, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
ax.set_title('Shot No.{}'.format(0))
plt.show()


print('tracecount = {}'.format(tracecount))
print('shotcount = {}'.format(shotcount))
print('dt = {} ms'.format(reader.dt))

fig,ax=plt.subplots(figsize=(5, 4))
ax.scatter(cog.recs[:, 0], cog.recs[:, 2], s=10, label='rec')
ax.scatter(cog.src[0], cog.src[2], s=5, label='src')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title('Shot No.{}'.format(0))
plt.show()

# dmap = get_resort_dict(filename)
# d = extract(filename, dmap)

# shot_no = 50
# vmin, vmax = np.percentile(d[shot_no], [1, 99])
# plt.figure(figsize=(5, 4))
# plt.imshow(d[shot_no], cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
# plt.colorbar()
# plt.title('Shot No.{}'.format(shot_no))
# plt.show()

# dt = read_dt(filename)
# print('dt = {} ms'.format(dt))
