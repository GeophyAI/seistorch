import torch
import numpy as np
import jax.numpy as jnp
from seistorch.signal import filter_jax, filter

class SeisArray:

    def __init__(self, data):
        self.data = data
        self.backend_name = None
        self.backend = self.__detect_backend__(data)

    def __detect_backend__(self, data):
        """Detect the backend of the data.

        Args:
            data (Any): The data to be detected.

        Raises:
            ValueError: If the data type is not supported.

        Returns:
            Target type: The backend of the data.
        """
        type_d = type(data)
        backend_house = {'jax': jnp, 'torch': torch, 'numpy': np}
        backend = None
        for key, value in backend_house.items():
            if key in str(type_d):
                backend = value
                self.backend_name = key
        if backend is None:
            raise ValueError(f"Unsupported data type {type_d}")
        return backend

    def envelope(self,):
        hilbert = self.hilbert()
        return self.backend.abs(hilbert)

    def hilbert(self,):
        nt, _, _ = self.data.shape #(nsamples, ntraces, nchannels)
        nfft = nt # the scipy implementation uses this

        # Compute the FFT
        data_fft = self.backend.fft.fft(self.data, nfft, 0)

        def gen_hilber_mask():
            h = np.zeros(nfft, dtype=np.float32)
            if nfft % 2 == 0:
                h[0] = h[nfft // 2] = 1
                h[1:nfft // 2] = 2
            else:
                h[0] = 1
                h[1:(nfft + 1) // 2] = 2
            
            h = np.expand_dims(h, 1)
            h = np.expand_dims(h, 2)

            if self.backend_name == 'torch':
                h = torch.from_numpy(h).to(self.data.device)
            if self.backend_name == 'jax':
                h = jnp.array(h)
            return h

        # Create the filter
        h = gen_hilber_mask()
        # Apply the filter and compute the inverse FFT
        hilbert_data = self.backend.fft.ifft(data_fft * h, None, 0)

        return hilbert_data # A complex number

    def filter(self, dt, freqs, order, axis):
        return filter_jax(self.data, dt, freqs, order, axis)
        