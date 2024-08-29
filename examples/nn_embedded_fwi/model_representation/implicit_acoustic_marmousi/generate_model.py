import numpy as np
import matplotlib.pyplot as plt
from configure import marmousi_path
import os
os.makedirs('models', exist_ok=True)

expand = 50
scale = 2
cfg_path = 'configure.py'

marmousi = np.load(marmousi_path)[:, expand:-expand]
marmousi = marmousi[::scale, ::scale]

with open(cfg_path, 'r') as file:
    lines = file.readlines()

with open(cfg_path, 'w') as f:
    for line in lines:
        if line.startswith('nx'):
            f.write('nx = %d\n' % marmousi.shape[1])
        elif line.startswith('nz'):
            f.write('nz = %d\n' % marmousi.shape[0])
        else:
            f.write(line)

plt.figure(figsize=(5, 3))
plt.imshow(marmousi, cmap='jet', aspect='auto')
plt.colorbar()

np.save('models/true.npy', marmousi)
