import numpy as np
import matplotlib.pyplot as plt

type1 = 'results_implicit_siren'
type2 = 'results_implicit_sirenscale'

for _type in [type1, type2]:
    loss = np.load(f"./{_type}/loss.npy")
    plt.plot(np.log10(loss.flatten())[:1000], label=_type)
plt.legend()
plt.show()