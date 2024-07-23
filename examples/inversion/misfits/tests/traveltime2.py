import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics.pairwise import cosine_similarity
import time

save_path = r"traveltime"
import os
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Func for Ricker
def ricker_wavelet(t, f):
    return (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)

# Func for calculating travel time
def calculate_travetime(signal1, signal2, dt=0.002):
    nt = signal2.size
    return (np.argmax(np.convolve(signal1, signal2))-nt)*dt

t1 = np.linspace(-1, 1, 1000, endpoint=False)
t2 = np.linspace(-0.5, 1.5, 1000, endpoint=False)
t3 = np.linspace(-1.5, 0.5, 1000, endpoint=False)

r1 = ricker_wavelet(t1, 10)
r2 = ricker_wavelet(t2, 10)*0.8
r3 = ricker_wavelet(t3, 10)*1.2

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t1, r1, label='ori', color='blue')
ax.plot(t1, r2, label='a', color='red', linestyle='dashed')
ax.plot(t1, r3, label='b', color='black', linestyle='dashed')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{save_path}/ricker.png", bbox_inches="tight", dpi=600)

# CC
cc1 = np.convolve(r1, r2)
cc2 = np.convolve(r1, r3)
delay = (np.arange(cc1.size)-r1.size)*abs(t1[1]-t1[0])
fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(delay, cc1, label='a', color='red', linestyle='dashed')
ax.plot(delay, cc2, label='b', color='black', linestyle='dashed')
ax.set_xlabel("Moved Time (s)")
ax.set_ylabel("Cross Correlation")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{save_path}/cc.png", bbox_inches="tight", dpi=600)

print(calculate_travetime(r1, r2))
print(calculate_travetime(r1, r3))