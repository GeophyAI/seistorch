import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shaowinw/default')
from pltparam import *
import string
from matplotlib.ticker import FormatStrFormatter
import torch
from scipy.ndimage import gaussian_filter1d

bwidth = 50

true = np.load('./velocity_model/true.npy')
init =  np.load('./velocity_model/init.npy')

path = [r'./results/l2/model_20.pt', 
        r'./results/traveltime/model_20.pt', 
        r'./results/traveltime_l2/model_30.pt']

titles = [r'Only FWI', 
         r'Tomography', 
         r'Tomography + FWI']
VP = []
#################### Show tomoed model
for path, title in zip(path, titles):
    vp = torch.load(path, 'cpu')['vp'].numpy()[bwidth:-bwidth, bwidth:-bwidth]
    VP.append(vp)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    vp = gaussian_filter1d(vp, 11, axis=1)
    # add colorbar
    dh = 20/1000.
    nz, nx = vp.shape
    extent = [0, nx*dh, nz*dh, 0]
    kwargs = dict(vmin=true.min(), 
                vmax=true.max(), 
                extent=extent,
                cmap='seismic',
                aspect='auto')
    cbar = plt.colorbar(ax.imshow(vp, **kwargs), ax=ax)
    ax.set_title(title)
    cbar.set_label('Vp (km/s)')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (km)')
    ax.xaxis.set_label_position('top')
    # keep a decimal in xticks
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    fig.savefig(f'figs/{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

trace_no = 400
fig, ax = plt.subplots(1,1, figsize=(5, 3))
depth = np.arange(nz)*20

ax.plot(depth, true[:,trace_no], label='True', c='black')
ax.plot(depth, init[:,trace_no], label='Initial', c='red')

for d, label in zip(VP, titles):
    ax.plot(depth, d[:,trace_no], label=title)
    # set xlabeltick bottom
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Vp (km/s)')
    ax.xaxis.set_label_position('bottom')

plt.legend()
plt.show()


