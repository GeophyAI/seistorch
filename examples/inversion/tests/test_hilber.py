import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('../../..')
from seistorch.transform import hilbert

def ricker(f, length, dt, peak_time):
    t = np.linspace(-peak_time/2, (length-1)*dt-peak_time/2, length)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return y

fm = 10
length = 1000
dt = 0.001
peak_time = 0.5

wavelet = ricker(fm, length, dt, peak_time)
plt.plot(wavelet, 'r')
plt.show()

hilbert_ricker = hilbert(torch.tensor(wavelet).float().unsqueeze(1).unsqueeze(2))
hilbert_ricker = hilbert_ricker.squeeze().numpy()
plt.plot(wavelet, 'black', label='Ricker')
plt.plot(hilbert_ricker.real, 'b', label='real')
plt.plot(hilbert_ricker.imag, 'g', label='imag')
plt.plot(np.abs(hilbert_ricker), 'r', label='abs')
plt.legend()
plt.show()

freq_amp = np.abs(np.fft.fft(wavelet))
freq_amp_hilbert = np.abs(np.fft.fft(hilbert_ricker))
freqs = np.fft.fftfreq(len(wavelet), dt)

plt.plot(freqs, freq_amp, 'black', label='Ricker')
plt.plot(freqs, freq_amp_hilbert, 'r', label='Hilbert')
plt.legend()
plt.show()