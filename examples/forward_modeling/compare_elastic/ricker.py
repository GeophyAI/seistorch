import numpy as np

ricker = np.loadtxt('ricker.txt')
np.save('wavelet.npy', ricker.astype(np.float32))

# Plot the wavelet
import matplotlib.pyplot as plt
plt.plot(ricker)
plt.show()