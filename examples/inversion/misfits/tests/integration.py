import numpy as np
import matplotlib.pyplot as plt
import lesio

savepath = r"cumulative"
import os
if not os.path.exists(savepath):
    os.mkdir(savepath)

def ricker_wavelet(t, f):
    return (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)


# 生成信号
t = np.linspace(-0.5, 1, 1000, endpoint=False)
frequency = 10  # 频率
original_signal = ricker_wavelet(t, frequency)

# first order integration
cum_signal = np.cumsum(original_signal)
cum_signal = cum_signal / np.max(np.abs(cum_signal))
# second order integration
cum_signal2 = np.cumsum(cum_signal)
cum_signal2 = cum_signal2 / np.max(np.abs(cum_signal2))

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t, original_signal, label="Original Signal")
ax.plot(t, cum_signal, label="1st Cumulative Signal")
ax.plot(t, cum_signal2, label="2nd Cumulative Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Cumulative Signal")
ax.legend(loc=4)
plt.tight_layout()
plt.show()
fig.savefig(f"{savepath}/cumulative.png", bbox_inches="tight", dpi=600)

amp_ori, freqs_ori = lesio.tools.freq_spectrum(original_signal.reshape(-1, 1), abs(t[1]-t[0]))
amp_cum1, freqs_cum1 = lesio.tools.freq_spectrum(cum_signal.reshape(-1, 1), abs(t[1]-t[0]))
amp_cum2, freqs_cum2 = lesio.tools.freq_spectrum(cum_signal2.reshape(-1, 1), abs(t[1]-t[0]))

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(freqs_ori[0:100], amp_ori[0:100], label="Original Signal")
ax.plot(freqs_cum1[0:100], amp_cum1[0:100], label="1st Cumulative Signal")
ax.plot(freqs_cum2[0:100], amp_cum2[0:100], label="2nd Cumulative Signal")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("Frequency Spectrum")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{savepath}/freq_spectrum_cumsum.png", bbox_inches="tight", dpi=600)

# High pass filter
hr = lesio.tools.fitler_fft(original_signal.reshape(-1, 1), abs(t[1]-t[0]), axis=0, N=4, low=5, mode='highpass').squeeze()
cum_hr = np.cumsum(hr)
cum_hr = cum_hr / np.max(np.abs(cum_hr))
cum_hr2 = np.cumsum(cum_hr)
cum_hr2 = cum_hr2 / np.max(np.abs(cum_hr2))

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(t, hr, label="High passed Ricker")
ax.plot(t, cum_hr, label="1st Cumulative High passed")
ax.plot(t, cum_hr2, label="2nd Cumulative High passed")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Cumulative High passed Signal")
ax.legend(loc=4)
plt.tight_layout()
plt.show()
fig.savefig(f"{savepath}/cumulative_highpass.png", bbox_inches="tight", dpi=600)


amp_hr, freqs_hr = lesio.tools.freq_spectrum(hr.reshape(-1, 1), abs(t[1]-t[0]))
amp_cum_hr, freqs_cum_hr = lesio.tools.freq_spectrum(cum_hr.reshape(-1, 1), abs(t[1]-t[0]))
amp_cum_hr2, freqs_cum_hr2 = lesio.tools.freq_spectrum(cum_hr2.reshape(-1, 1), abs(t[1]-t[0]))

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(freqs_hr[0:100], amp_hr[0:100], label="High passed Ricker")
ax.plot(freqs_cum_hr[0:100], amp_cum_hr[0:100], label="1st Cumulative High passed")
ax.plot(freqs_cum_hr2[0:100], amp_cum_hr2[0:100], label="2nd Cumulative High passed")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("Frequency Spectrum")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f"{savepath}/freq_spectrum_cumsum_highpass.png", bbox_inches="tight", dpi=600)