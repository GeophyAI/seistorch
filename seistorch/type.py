import torch
from seistorch.utils import to_tensor

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

    def numpy(self):
        for i in range(len(self.data)):
            if isinstance(self.data[i], torch.Tensor):
                self.data[i] = self.data[i].detach().cpu().numpy()
        return self
    
    def has_nan(self):
        for i in range(len(self.data)):
            if isinstance(self.data[i], torch.Tensor):
                if torch.isnan(self.data[i]).any():
                    raise ValueError("The tensor list contains NaN values.")
        return False

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return str(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
