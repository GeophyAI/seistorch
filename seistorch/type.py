import torch
import numpy as np
from seistorch.utils import to_tensor
from typing import Any
HANDLED_FUNCTIONS = {}

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


class Coordinate:

    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self) -> Any:
        return self._x
    
    @property
    def y(self) -> Any:
        return self._y
    
    @property
    def z(self) -> Any:
        return self._z

    def __str__(self):
        return 'x: {}, y: {}, z: {}'.format(self.x, self.y, self.z)
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Coordinate):
            return self.x == __value.x and self.y == __value.y and self.z == __value.z
        else:
            return False
        
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

class Trace:

    def __init__(self, src: Coordinate, rec: Coordinate, idx_in_segy: int = None):
        self._src = src
        self._rec = rec
        self._idx_in_segy = idx_in_segy

    @property
    def src(self):
        return (self._src.x, self._src.y, self._src.z)
    
    @property
    def rec(self):
        return (self._rec.x, self._rec.y, self._rec.z)
    
    @property
    def idx_in_segy(self):
        return self._idx_in_segy
    
class CommonShotGather:

    def __init__(self, src: Coordinate, recs: list, data: np.ndarray):
        self._src = np.asarray(src)
        self._recs = np.asarray(recs)
        self._data = data

    @property
    def src(self):
        return self._src
    
    @property
    def recs(self):
        return self._recs
    
    @property
    def data(self):
        return self._data
    
    
