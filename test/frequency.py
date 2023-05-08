import torch
import numpy as np

# Problem domain parameters
nx, nz = 101, 101
dx, dz = 10, 10
nt = 1000
dt = 0.001

# Model parameters
vp = torch.ones(nx, nz) * 2000

# Source and receiver locations
src_x, src_z = nx // 2, nz // 2
rec_x, rec_z = nx // 2, 10

# Frequency-domain source wavelet
freqs = np.fft.rfftfreq(nt, d=dt)
src_wavelet = torch.tensor(np.exp(-1j * 2 * np.pi * freqs * nt), dtype=torch.cfloat)

# Define the wave equation in the frequency domain using PyTorch
def wave_equation_freq(vp, src_wavelet, freqs, src_x, src_z, rec_x, rec_z):
    nx, nz = vp.shape
    kx = torch.fft.rfftfreq(nx, d=dx) * 2 * np.pi
    kz = torch.fft.rfftfreq(nz, d=dz) * 2 * np.pi
    Kx, Kz, Freqs = torch.meshgrid(kx, kz, freqs)

    # Compute the wavenumber squared tensor
    k_squared = Kx**2 + Kz**2

    # Compute the Green's function for each frequency
    greens_func = torch.exp(-1j * torch.sqrt(Freqs**2 / vp[src_x, src_z]**2 - k_squared))

    # Multiply the Green's function by the source wavelet
    wavefield_freq = greens_func * src_wavelet.view(1, 1, -1)

    # Inverse Fourier transform to obtain the time-domain wavefield
    wavefield_time = torch.fft.irfft(wavefield_freq, n=nt, dim=2)

    # Extract the receiver data
    rec_data = wavefield_time[rec_x, rec_z, :]

    return rec_data

# Perform forward modeling
rec_data = wave_equation_freq(vp, src_wavelet, freqs, src_x, src_z, rec_x, rec_z)
