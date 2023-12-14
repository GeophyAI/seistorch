import numpy as np
import matplotlib.pyplot as plt

type1 = 'siren'
type2 = 'siren_scale'

for _type in [type1, type2]:
    loss = np.load(f"./{_type}/loss.npy")
    plt.plot(np.log10(loss.flatten())[:2000], label=_type)
plt.legend()
plt.show()