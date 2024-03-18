from typing import Any
import segyio
import numpy as np
import matplotlib.pyplot as plt

class SegyReader:

    def __init__(self, path, analysis=False):
        self.path = path
        self.trace_srcs, self.recs = self.get_source_receiver_coordinates()
        self.dmap = self.map2csg()

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
    
    def get_source_receiver_coordinates(self, ):
        self._open()
        source_coordinates = np.column_stack((self.f.attributes(segyio.TraceField.SourceX),
                                              self.f.attributes(segyio.TraceField.SourceY), 
                                              self.f.attributes(segyio.TraceField.SourceDepth)))

        receiver_coordinates = np.column_stack((self.f.attributes(segyio.TraceField.GroupX),
                                                self.f.attributes(segyio.TraceField.GroupY), 
                                                self.f.attributes(segyio.TraceField.ReceiverGroupElevation)))
        self._close()
        return source_coordinates, receiver_coordinates        

    def map2csg(self, ):
        """Sort the segy file by the coordinate of the source and receiver"""

        self.unique_srcs, unique_indices, counts = np.unique(self.trace_srcs, axis=0, return_inverse=True, return_counts=True)

        indices_grouped = np.split(np.argsort(unique_indices), np.cumsum(counts))[:-1]

        recs_grouped = [self.recs[indices] for indices in indices_grouped]

        dmap = dict(zip(map(tuple, self.unique_srcs), 
                        [{'idx_in_segy': np.asarray(indices), 
                          'recs': np.stack(recs_values, axis=-1)} for indices, recs_values in zip(indices_grouped, recs_grouped)]))

        return dmap
    
    def get_shot(self, shot_no):
        """Get the data of shot gather by shot_no"""
        if shot_no >= len(self.dmap):
            raise ValueError('shot_no should be less than {}'.format(len(self.dmap)))
        if shot_no < 0:
            shot_no = len(self.dmap) + shot_no
        self._open()
        src = tuple(self.unique_srcs[shot_no])
        rec = self.dmap[src]['recs'].T
        idx_in_segy = self.dmap[src]['idx_in_segy']
        sorted_data = sorted(zip(idx_in_segy, rec), key=lambda x: x[0])
        idx_in_segy, rec = zip(*sorted_data)
        
        data = []
        for i in idx_in_segy:
            data.append(self.f.trace.raw[i])
        self._close()
        data = np.array(data).T
        return src, rec, data


filename = '/home/wangsw/wangsw/model/1997_2.5D_shots.segy'

reader = SegyReader(filename)

shot_no = 200

src, rec, data = reader.get_shot(shot_no)
vmin, vmax = np.percentile(data, [1, 99])
plt.imshow(data, cmap='seismic', vmin=vmin, vmax=vmax)
plt.show()
plt.plot(np.array(rec)[:,0], 'r*')
plt.show()