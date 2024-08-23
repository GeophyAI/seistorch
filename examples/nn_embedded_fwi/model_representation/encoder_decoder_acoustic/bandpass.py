from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from configure import dt, fm, delay, nt
from utils_torch import ricker

def highpass(data, dt=0.001, freqs=[2,5], forder=3, btype='highpass', axis=1):
      
    wn = [2*freq/(1/dt) for freq in list(freqs)]
    wn = wn[0] if len(wn)==1 else wn

    # call _filter_
    b, a = signal.butter(forder, Wn=wn, btype=btype)

    filtered_data = signal.filtfilt(b, a, data, axis=axis).astype(np.float32)
    return filtered_data


def show_freq_spectrum(data, dt=0.001, end_freq=25, title='Frequency Spectrum'):
    plt.figure(figsize=(5, 3))
    freqs = np.fft.fftfreq(data.shape[1], dt)
    amp = np.sum(np.abs(np.fft.fft(data, axis=1)), axis=(0,2))
    freqs = freqs[:len(freqs)//2]
    amp = amp[:len(amp)//2]
    amp = amp[freqs<end_freq]
    freqs = freqs[freqs<end_freq]
    plt.plot(freqs, amp)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()

obs = np.load("obs.npy")
wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
wave = wave.reshape(1, nt, 1).cpu().numpy()

show_freq_spectrum(obs, dt=dt, title='Frequency Spectrum of Observed Data')

filtered_data = highpass(obs, dt=dt, freqs=[5], forder=3, btype='highpass', axis=1)
filtered_wave = highpass(wave, dt=dt, freqs=[5], forder=3, btype='highpass', axis=1)
filtered_wave = filtered_wave.squeeze()

show_freq_spectrum(filtered_data, dt=dt, title='Frequency Spectrum of Filtered Data')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
vmin, vmax = np.percentile(obs[0], [1, 99])
plt.imshow(obs[0], vmin=vmin, vmax=vmax, aspect='auto', cmap='seismic')
plt.title('Observed Data')
plt.subplot(1, 2, 2)
vmin, vmax = np.percentile(filtered_data[0], [1, 99])
plt.imshow(filtered_data[0], vmin=vmin, vmax=vmax, aspect='auto', cmap='seismic')
plt.title('Filtered Data')
plt.show()

plt.plot(wave[0], label='Original Wavelet')
plt.plot(filtered_wave, label='Filtered Wavelet')
plt.legend()
plt.show()

np.save("obs_filtered.npy", filtered_data)
np.save("wave_filtered.npy", filtered_wave)
