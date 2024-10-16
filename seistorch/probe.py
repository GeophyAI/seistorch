import torch

from .utils import to_tensor


class WaveProbe(torch.nn.Module):
	def __init__(self, batchidx=None, reccounts=[], **kwargs):
		super().__init__()
		self._ndim = len(kwargs)
		self.coord_labels = list(kwargs.keys())
		# Register index buffer
		for key, value in kwargs.items():
			self.register_buffer(key, to_tensor(value, dtype=torch.int64))
		self.forward = self.get_forward_func()
		self.batchsize = self.x.size(0) if self.x.ndim>1 else 1
  
		self.bidx = batchidx

		# This parameter is used for split the records in RNN
		# DONNOT DELETE IT
		self.reccounts = reccounts 
  
	@property
	def ndim(self,):
		return self._ndim
	
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
     

	def forward2d(self, x):

		return x[self.bidx, self.y, self.x]
	
	def forward3d(self, x):

  		return x[self.bidx, self.x, self.z, self.y]

class WaveIntensityProbe(WaveProbe):
	def __init__(self, **kwargs):
		#print('WaveIntensityProbe', x, y)
		super().__init__(**kwargs)

	def forward(self, x):
		return super().forward(x)#.pow(2)
