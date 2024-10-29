import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time, sys, importlib
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from utils_jax import *

def main():

    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file.py>")
        sys.exit(1)

    config_file = sys.argv[1].replace(".py", "")

    try:
        config = importlib.import_module(config_file)
    except ModuleNotFoundError:
        print(f"Config file {config_file}.py not found!")
        sys.exit(1)

    globals().update(vars(config))

    # Load velocity
    true = np.load(true_path)/unit
    true = true[::model_scale_z, ::model_scale_x]

    print('Standard deviation of the velocity model:', true.std())
    print('Mean of the velocity model:', true.mean())
    print('Shape of the velocity model:', true.shape)
    print(f'Size of the velocity model: {true.shape[0]*dh}m x {true.shape[1]*dh}m')

    ori_domain = true.shape
    ori_nz, ori_nx = ori_domain
    if not multiple:
        padding = ((pmln, pmln), (pmln, pmln))
        padding_domain = (ori_nz + 2*pmln, ori_nx + 2*pmln)
    else:
        padding = ((0, pmln), (pmln, pmln))
        padding_domain = (ori_nz + pmln, ori_nx + 2*pmln)

    pmlc = generate_pml_coefficients_2d(padding_domain, N=pmln, multiple=multiple).numpy()
    pmlc = jnp.array(pmlc)
    plt.imshow(pmlc, cmap="jet", aspect="auto")
    plt.colorbar()
    plt.savefig("pmlc.png", dpi=300, bbox_inches="tight")
    plt.close()

    wave = ricker(jnp.arange(nt) * dt - delay * dt, f=fm)
    tt = np.arange(nt) * dt
    plt.plot(tt, wave)
    plt.title("Wavelet")
    plt.tight_layout()
    plt.savefig("wavelet.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Frequency spectrum
    freqs = np.fft.fftfreq(nt, dt)[:nt//2]
    amp = np.abs(np.fft.fft(wave))[:nt//2]
    amp = amp[freqs <= 20] 
    freqs = freqs[freqs <= 20]
    plt.plot(freqs, amp)
    plt.title("Frequency spectrum")
    plt.savefig("wavelet_freq_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Geometry
    srcxs = np.arange(expand+pmln, padding_domain[1]-expand-pmln, srcx_step).tolist()
    srczs = (np.ones_like(srcxs) * srcz).tolist()
    src_loc = list(zip(srcxs, srczs))

    recxs = np.arange(expand+pmln, padding_domain[1]-expand-pmln, 1).tolist()
    reczs = (np.ones_like(recxs) * recz).tolist()
    rec_loc = list(zip(recxs, reczs))

    # Show geometry
    print(f"The number of sources: {len(src_loc)}")
    print(f"The number of receivers: {len(rec_loc)}")
    kwargs = dict(b=pmlc, domain=padding_domain, dt=dt, h=dh, recz=recz, pmln=pmln, spatial_order=spatial_order)
    start_time = time.time()
    rec_obs = forward(wave, jnp.pad(true, padding, mode='edge'), src_list=np.array(src_loc), **kwargs)
    end_time = time.time()
    print(f"Forward modeling time: {end_time - start_time:.2f}s")
    show_gathers(rec_obs, figsize=(10, 6), save_path='.')

    np.save(rec_obs_path, rec_obs)

if __name__ == "__main__":
    main()