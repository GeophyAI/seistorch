import numpy as np
import matplotlib.pyplot as plt

obs = np.load('./results/fwi_classic_ADHABC/obs0.npy')
syn = np.load('./results/fwi_classic_ADHABC/syn0.npy')

plt.imshow(obs[0], aspect='auto')
plt.show()
plt.imshow(syn[0], aspect='auto')
plt.show()