import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

savepath = r"env_plus_ig"
import os
if not os.path.exists(savepath):
    os.mkdir(savepath)

def ricker_wavelet(t, f):
    return (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)


# 生成信号
t = np.linspace(-0.5, 1, 1000, endpoint=False)
frequency = 10  # 频率
original_signal = ricker_wavelet(t, frequency)

env = np.abs(hilbert(original_signal))
ig = np.cumsum(env)
# Original
fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t, original_signal, label="Original Signal", c='red')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{savepath}/original.png", bbox_inches="tight", dpi=600)

# Original + env
fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t, original_signal, label="Original Signal", c='red')
ax.plot(t, env, label="Envelope", c='black')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{savepath}/Envelope.png", bbox_inches="tight", dpi=600)


# Env+ig
fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t, env, label="Envelope", c='black')
ax.plot(t, ig/ig.max(), label="Integration", c='blue')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{savepath}/Integration.png", bbox_inches="tight", dpi=600)