import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import lesio

save_path = r"envelope"
import os
if not os.path.exists(save_path):
    os.mkdir(save_path)

def ricker_wavelet(t, f):
    return (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)

frequency=10
t = np.linspace(-0.5, 1, 1000, endpoint=False)
r = ricker_wavelet(t, frequency)
env = np.abs(hilbert(r))

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t, r, label="Ricker")
ax.plot(t, env, label="Envelope")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Envelope")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{save_path}/envelope.png", bbox_inches="tight", dpi=600)

amp_r, freqs_r = lesio.tools.freq_spectrum(r.reshape(-1, 1), abs(t[1]-t[0]))
amp_env, freqs_env = lesio.tools.freq_spectrum(env.reshape(-1, 1), abs(t[1]-t[0]))
fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(freqs_r[0:100], amp_r[0:100], label="Ricker")
ax.plot(freqs_env[0:100], amp_env[0:100], label="Envelope")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("Frequency Spectrum")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{save_path}/freq_spectrum_envelope.png", bbox_inches="tight", dpi=600)

# High pass filter
hr = lesio.tools.fitler_fft(r.reshape(-1, 1), abs(t[1]-t[0]), axis=0, low=5, mode='highpass').squeeze()

env_hr = np.abs(hilbert(hr))

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t, r, label="Ricker")
ax.plot(t, hr, label="High passed Ricker")
ax.plot(t, env_hr, label="Envelope of high passed")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Envelope of high passed Ricker")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{save_path}/highpass.png", bbox_inches="tight", dpi=600)


amp_hr, freqs_hr = lesio.tools.freq_spectrum(hr.reshape(-1, 1), abs(t[1]-t[0]))
amp_env_hr, freqs_env_hr = lesio.tools.freq_spectrum(env_hr.reshape(-1, 1), abs(t[1]-t[0]))

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(freqs_r[0:100], amp_r[0:100], label="Ricker")
ax.plot(freqs_hr[0:100], amp_hr[0:100], label="High passed Ricker")
ax.plot(freqs_env_hr[0:100], amp_env_hr[0:100], label="Envelope of high passed")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("Frequency Spectrum")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{save_path}/freq_spectrum_envelope_highpass.png", bbox_inches="tight", dpi=600)