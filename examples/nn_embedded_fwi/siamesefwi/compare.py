import numpy as np
import matplotlib.pyplot as plt

v_classic = np.load('inverted_by_classic.npy')
v_siamese = np.load('inverted_by_vggloss.npy')
titles = ['Inverted by classic FWI', 'Inverted by VGG FWI']
vmin, vmax=1500, 5500
plt.figure(figsize=(10, 5))
for i, v in enumerate([v_classic, v_siamese]):
    plt.subplot(1, 2, i+1)
    plt.imshow(v, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
    plt.colorbar()
    plt.title(titles[i])

merror_classic = np.load('model_error_by_classic.npy')
merror_siamese = np.load('model_error_by_implicitloss.npy')
plt.figure(figsize=(5, 3))
plt.plot(merror_classic, label='Classic FWI')
plt.plot(merror_siamese, label='VGG FWI')
plt.xlabel('Epoch')
plt.ylabel('Model Error')
plt.legend()
plt.show()
