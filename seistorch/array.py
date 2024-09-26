import torch
import numpy as np
import jax.numpy as jnp
from seistorch.signal import filter_jax, filter
from seistorch.utils import to_tensor
HANDLED_FUNCTIONS = {}

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

class TensorList(list):

    """A list of torch.Tensors"""

    def __init__(self, input_list=[]):
        self.data = []
        for item in input_list:
            item = item if isinstance(item, torch.Tensor) else to_tensor(item)
            self.data.append(item)

    @property
    def device(self,):
        return self.data[0].device

    @property
    def shape(self,):
        return (len(self.data), )

    def append(self, item):
        item = item if isinstance(item, torch.Tensor) else to_tensor(item)
        self.data.append(item)

    def cuda(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].cuda()
        return self
    
    # def extend(self, iterable):
    #     for item in iterable:
    #         item = item if isinstance(item, torch.Tensor) else to_tensor(item)
    #         if item.ndim == 2:
    #             item = item.unsqueeze(2)
    #         self.append(item)
    
    def has_nan(self):
        for i in range(len(self.data)):
            if isinstance(self.data[i], torch.Tensor):
                if torch.isnan(self.data[i]).any():
                    raise ValueError("The tensor list contains NaN values.")
        return False

    def numpy(self):
        for i in range(len(self.data)):
            if isinstance(self.data[i], torch.Tensor):
                self.data[i] = self.data[i].detach().cpu().numpy()
        return self
    
    def stack(self):
        max_shape = max([tensor.shape for tensor in self.data])
        padded_tensors = [torch.nn.functional.pad(tensor, (0, max_shape[1] - tensor.shape[1], 0, max_shape[0] - tensor.shape[0])) for tensor in self.data]
        return torch.stack(padded_tensors, dim=0)

    def tensor(self,):
        return self.data

    def to(self, device):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].to(device)
        return self
    
    def tolist(self,):
        return self

    def __getitem__(self, index):
        return self.data[index]
    
    def __iter__(self):
        return iter(self.data)

    def __mul__(self, other):
        if isinstance(other, TensorList) and len(self.data) == len(other.data):
            result = TensorList()
            for i in range(len(self.data)):
                if isinstance(self.data[i], torch.Tensor) and isinstance(other.data[i], torch.Tensor):
                    result.append(self.data[i] * other.data[i])
                else:
                    raise ValueError("Multiplication is only defined for instances of TensorList containing torch.Tensors.")
            return result
        else:
            raise ValueError("Multiplication is only defined between two instances of TensorList with the same length.")

    def __pow__(self, exponent):
        
        for i in range(len(self.data)):
            if isinstance(self.data[i], torch.Tensor):
                self.data[i] = self.data[i]**exponent

        return self

    def __str__(self):
        return str(self.data)      