import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
sys.path.append('../../../../')
from seistorch.transform import envelope
from seistorch.signal import ricker_wave, filter

os.makedirs('figures', exist_ok=True)

def freq_spectrum(d, dt, end_Freq=25):
    freqs = np.fft.fftfreq(d.shape[0], dt)
    amp = np.abs(np.fft.fft(d))
    freqs = freqs[:len(freqs)//2]
    amp = amp[:len(amp)//2]
    amp = amp[freqs<end_Freq]
    freqs = freqs[freqs<end_Freq]
    return freqs, amp


# Generate a Ricker wavelet
fm = 10 # dominant frequency
dt = 0.001 # time step
nt = 1000 # time samples
delay = 200 # delay in samples
t = np.arange(nt)*dt
ricker = ricker_wave(fm, dt, nt, delay)

# Compute the envelope of the Ricker wavelet
ricker = ricker.reshape(-1, 1, 1)
ricker_envelope = envelope(ricker)
plt.figure(figsize=(5, 3))
plt.plot(t, ricker_envelope.squeeze(), 'b', label='Envelope')
plt.plot(t, ricker.squeeze(), 'r', label='Ricker')
plt.xlabel('Time (s)')
plt.title('Ricker wavelet and its envelope')
plt.legend()
plt.tight_layout()
plt.savefig('figures/ricker_envelope.png', dpi=300, bbox_inches='tight')
plt.show()

# Show spectrum
ricker = ricker.numpy().squeeze()
ricker_envelope = ricker_envelope.numpy().squeeze()
end_freq=50
freqs, amp_ricker = freq_spectrum(ricker, dt, end_Freq=end_freq)
freqs, amp_envelope = freq_spectrum(ricker_envelope, dt, end_Freq=end_freq)
plt.figure(figsize=(5, 3))
plt.plot(freqs, amp_ricker, 'b', label='Ricker')
plt.plot(freqs, amp_envelope, 'r', label='Envelope')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title('Frequency Spectrum')
plt.legend()
plt.tight_layout()
plt.savefig('figures/ricker_envelope_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()

# High pass filter
highpass_ricker = filter(ricker, dt, freqs=[5], forder=4, btype='highpass', axis=0)
highpass_ricker = torch.from_numpy(highpass_ricker)
env_highpass_ricker = envelope(highpass_ricker.reshape(-1, 1, 1)).numpy().squeeze()
highpass_ricker = highpass_ricker.numpy().squeeze()

plt.figure(figsize=(5, 3))
plt.figure(figsize=(5, 3))
plt.plot(t, highpass_ricker.squeeze(), 'b', label='Envelope')
plt.plot(t, env_highpass_ricker.squeeze(), 'r', label='Ricker')
plt.xlabel('Time (s)')
plt.title('Ricker wavelet and its envelope')
plt.legend()
plt.tight_layout()
plt.savefig('figures/highpass_ricker_envelope.png', dpi=300, bbox_inches='tight')
plt.show()

# Show spectrum
freqs, amp_ricker = freq_spectrum(highpass_ricker, dt, end_Freq=end_freq)
freqs, amp_envelope = freq_spectrum(env_highpass_ricker, dt, end_Freq=end_freq)
plt.figure(figsize=(5, 3))
plt.plot(freqs, amp_ricker, 'b', label='Ricker')
plt.plot(freqs, amp_envelope, 'r', label='Envelope')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title('Frequency Spectrum')
plt.legend()
plt.tight_layout()
plt.savefig('figures/highpass_ricker_envelope_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()




