import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle   

def generate_pml_coefficients_2d(domain_shape, N=50, B=100., multiple=False):
    Nx, Ny = domain_shape

    R = 10**(-((np.log10(N)-1)/np.log10(2))-3)
    #d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
    R = 1e-6; order = 2; cp = 1000.
    d0 = (1.5*cp/N)*np.log10(R**-1)
    d_vals = d0 * torch.linspace(0.0, 1.0, N + 1) ** order
    d_vals = torch.flip(d_vals, [0])

    d_x = torch.zeros(Ny, Nx)
    d_y = torch.zeros(Ny, Nx)
    
    if N > 0:
        d_x[0:N + 1, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
        d_x[(Ny - N - 1):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
        if not multiple:
            d_y[:, 0:N + 1] = d_vals.repeat(Ny, 1)
        d_y[:, (Nx - N - 1):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

    _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)

    return _d

def imshow(data, vmin=None, vmax=None, cmap=None, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.show()

def show_gathers(rec, size=3, figsize=(8, 5)):
    randno = np.random.randint(0, rec.shape[0], size=size)
    fig,axes=plt.subplots(1, randno.shape[0], figsize=figsize)

    if size==1:
        axes=[axes]

    for i, ax in enumerate(axes):
        vmin,vmax=np.percentile(rec[i], [1, 99])
        kwargs=dict(vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.imshow(rec[randno[i]], **kwargs)
        ax.set_title(f"shot {randno[i]}")
    plt.tight_layout()
    plt.show()

def showgeom(vel, src_loc, rec_loc, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(vel, vmin=vel.min(), vmax=vel.max(), cmap="seismic", aspect="auto")
    plt.colorbar()
    plt.scatter(*zip(*src_loc), c="r", marker="v", s=100, label="src")
    plt.scatter(*zip(*rec_loc), c="b", marker="^", s=10, label="rec")
    plt.legend()
    plt.show()

def show_freq_spectrum(data, dt=0.001, end_freq=25, title='Frequency Spectrum'):
    plt.figure(figsize=(5, 3))
    freqs = np.fft.fftfreq(data.shape[0], dt)
    amp = np.sum(np.abs(np.fft.fft(data, axis=0)), axis=(1,2))
    freqs = freqs[:len(freqs)//2]
    amp = amp[:len(amp)//2]
    amp = amp[freqs<end_freq]
    freqs = freqs[freqs<end_freq]
    plt.plot(freqs, amp)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()

def to_tensor(data, device):
    return torch.from_numpy(data).float().to(device)

def ricker(t, f=10):
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return torch.from_numpy(r).float()

def gradient(input, dim=-1, forward=True, padding_value=0):

    def forward_diff(x, dim=-1, padding_value=0):
        """
        Compute the forward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The forward difference of the input tensor.
        """
        # x[:,0] = padding_value
        diff = x - torch.roll(x, shifts=1, dims=dim)
        if dim == 1:
            diff[:, 0] = padding_value
        elif dim == 2:
            diff[..., 0] = padding_value  # pad with specified value
        return diff

    def backward_diff(x, dim=-1, padding_value=0):
        """
        Compute the backward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The backward difference of the input tensor.
        """
        # x[...,-1] = padding_value
        diff = torch.roll(x, shifts=-1, dims=dim) - x
        if dim == 1:
            diff[:, -1] = padding_value
        elif dim == 2:
            diff[..., -1] = padding_value  # pad with specified value
        return diff

    if forward:
        return forward_diff(input, dim=dim)
    else:
        return backward_diff(input, dim=dim)

def step(parameters, wavefields, geometry):

    vp, vs, rho = parameters
    vx, vz, txx, tzz, txz = wavefields
    dt, h, d = geometry

    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))
    c = 0.5*dt*d

    vx_x = gradient(vx, 2)
    vz_z = gradient(vz, 1, False)
    vx_z = gradient(vx, 1)
    vz_x = gradient(vz, 2, False)

    # Equation A-8
    y_txx  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    # Equation A-9
    y_tzz  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    # Equation A-10
    y_txz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = gradient(y_txx, 2, False)
    txz_z = gradient(y_txz, 1, False)
    tzz_z = gradient(y_tzz, 1)
    txz_x = gradient(y_txz, 2)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz

def forward(wave, parameters, pmlc, src_list, domain, dt, h, dev, npml=50, recz=0):
    
    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)

    dt = torch.tensor(dt, dtype=torch.float32, device=dev)
    h = torch.tensor(h, dtype=torch.float32, device=dev)

    vx = torch.zeros(nshots, *domain, device=dev)
    vz = torch.zeros(nshots, *domain, device=dev)
    txx = torch.zeros(nshots, *domain, device=dev)
    tzz = torch.zeros(nshots, *domain, device=dev)
    txz = torch.zeros(nshots, *domain, device=dev)

    wavefields = [vx, vz, txx, tzz, txz]
    geoms = [dt, h, pmlc]

    rec = torch.zeros(nshots, nt, nx-2*npml).to(dev)

    shots = torch.arange(nshots).to(dev)
    srcx, srcz = zip(*src_list)
    src_mask = torch.zeros_like(vx)
    src_mask[shots, srcz, srcx] = 1

    for it in range(nt):
        # GPU ALIGNED 
        wavefields[1] += src_mask * wave[it]
        wavefields = step(parameters, wavefields, geoms)
        wavefields = list(wavefields)
        rec[:,it, :] = wavefields[1][:, recz, npml:-npml]
    return rec

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)