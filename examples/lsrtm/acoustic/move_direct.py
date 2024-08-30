import numpy as np
import matplotlib.pyplot as plt
import h5py

def read_h5py(filename, shot_no):
    with h5py.File(filename, 'r') as f:
        data = f[f'shot_{shot_no}'][:]
    return data

def write_h5py(filename, data, shot_no):
    with h5py.File(filename, 'a') as f:
        f.create_dataset(f'shot_{shot_no}', data=data)
    
savepath = r'moved_direct_wave.hdf5'
with h5py.File(savepath, 'w') as f:
    pass

for i in range(93):
    obs = read_h5py('obs.hdf5', i)
    direct = read_h5py('obs_direct.hdf5', i)
    _obs = obs - direct
    write_h5py(savepath, _obs, i)
    # fig,axes=plt.subplots(1,3,figsize=(12,4))
    # vmin, vmax = np.percentile(obs, [1, 99])
    # kwargs = dict(cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    # axes[0].imshow(obs, **kwargs)
    # axes[0].set_title('Observed')
    # axes[1].imshow(direct, **kwargs)
    # axes[1].set_title('Direct')
    # axes[2].imshow(_obs, **kwargs)
    # axes[2].set_title('Moveout')
    # plt.tight_layout()
    # plt.show()

    # break

print('Done')