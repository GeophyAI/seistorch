import torch

from .utils import to_tensor


class WaveProbe(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self._ndim = len(kwargs)
		self.coord_labels = list(kwargs.keys())
		# Register index buffer
		for key, value in kwargs.items():
			self.register_buffer(key, to_tensor(value, dtype=torch.int64))

		self.forward = self.get_forward_func()

	@property
	def ndim(self,):
		return self._ndim
	
	def get_forward_func(self, ):
		return getattr(self, f"forward{self.ndim}d")

	def forward2d(self, x):
		return x[:, self.y, self.x]
	
	def forward3d(self, x):
		return x[:, self.x, self.z, self.y]

class WaveIntensityProbe(WaveProbe):
	def __init__(self, **kwargs):
		#print('WaveIntensityProbe', x, y)
		super().__init__(**kwargs)

	def forward(self, x):
		return super().forward(x)#.pow(2)
