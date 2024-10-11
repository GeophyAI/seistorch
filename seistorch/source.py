import torch
import inspect
from .utils import to_tensor
import jax.numpy as jnp

class WaveSourceBase:

    def __init__(self, bidx=None, second_order_equation=False, sharding=None, **kwargs):
        super().__init__()
        self._ndim = len(kwargs)
        self.coord_labels = list(kwargs.keys())
        self.bidx = bidx
        self.sharding = sharding
        # self.forward = self.get_forward_func()
        self._source_encoding=False

    @property
    def ndim(self,):
        return self._ndim
    
    @property
    def source_encoding(self,):
        return self._source_encoding
    
    @source_encoding.setter
    def source_encoding(self, value):
        self._source_encoding = value

    def coords(self,):
        """Return the coordinates of the source.

        Returns:
            dict: A list of coordinates.
            Example: {'x': [x1, x2, ..., xn], 
                      'y': [y1, y2, ..., yn], 
                      'z': [z1, z2, ..., zn]]}
        """
        return dict(zip(self.coord_labels, [getattr(self, key) for key in self.coord_labels]))

    def get_forward_func(self, ):
        return getattr(self, f"forward{self.ndim}d")

    def forward2d(self, Y, X, f=1.):
        
        if not self.source_encoding:
            # Y += self.smask * f*X
            return Y + self.smask * f*X

        if self.source_encoding:
            Y[..., self.y, self.x] += f*X
            return Y

    def forward3d(self, Y, X, f=1.):
        if not self.source_encoding:
            # Y += self.smask * f*X
            return Y + self.smask * f*X

        if self.source_encoding:
            Y[..., self.z, self.y, self.x] += f*X
            return Y

    def forward(self, Y, X, f=1.):
        return self.get_forward_func()(Y, X, f)

class WaveSourceTorch(WaveSourceBase, torch.nn.Module):

    def __init__(self, bidx=None, second_order_equation=False, sharding=None, **kwargs):
        torch.nn.Module.__init__(self)
        super().__init__(bidx, second_order_equation, sharding=None, **kwargs)

        for key, value in kwargs.items():
            value = None if value is None else to_tensor(value, dtype=torch.int64)
            self.register_buffer(key, value)

    def forward(self, Y, X, f=1.):
        Y_new = Y.clone() if self.second_order_equation else Y
        return super().forward(Y_new, X, f)

class WaveSourceJax(WaveSourceBase):
    
    def __init__(self, bidx=None, second_order_equation=False, sharding=None, **kwargs):

        super().__init__(bidx, second_order_equation, sharding=sharding, **kwargs)

        for key, value in kwargs.items():
            setattr(self, key, jnp.array(value, dtype=jnp.int32))

    def forward_jax(self, Y, X, f=1.):
        Y = Y.at[..., self.y, self.x].add(f*X)
        return Y

    def __call__(self, *args, **kwargs):
        if not self.source_encoding:
            return super().forward(*args, **kwargs)
        else:
            return self.forward_jax(*args, **kwargs)