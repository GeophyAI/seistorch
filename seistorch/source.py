#import skimage
import torch
import inspect
from .utils import to_tensor

class WaveSource(torch.nn.Module):
	def __init__(self, bidx=None, second_order_equation=False, **kwargs):

		super().__init__()
		self._ndim = len(kwargs)
		self.coord_labels = list(kwargs.keys())

		for key, value in kwargs.items():
			value = None if value is None else to_tensor(value, dtype=torch.int64)
			self.register_buffer(key, value)
   
		self.bidx = bidx
		self.forward = self.get_forward_func()
		self._source_encoding=False
		self.second_order_equation = second_order_equation
  
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

	def forward2d(self, Y, X, dt=1.0):
		Y_new = Y.clone() if self.second_order_equation else Y
		if not self.source_encoding:
			for idx in range(self.x.size(0)):
				Y_new[idx:idx+1, self.y[idx]:self.y[idx]+1, self.x[idx]] += dt*X
		if self.source_encoding:
			Y_new[..., self.y, self.x] += dt*X
		return Y_new

	def forward3d(self, Y, X, dt=1.0):
		Y_new = Y.clone()

		# for idx in range(self.x.size(0)):
		# 	Y_new[idx:idx+1, self.x[idx], self.z[idx], self.y[idx]] += dt*X[0]
		if not self.source_encoding:
			for idx in range(self.x.size(0)):
				Y_new[idx:idx+1, self.x[idx]:self.x[idx]+1, self.z[idx], self.y[idx]] += dt*X

		#Y_new[self.bidx, self.x, self.z, self.y] = Y_new[self.bidx, self.x, self.z, self.y]+dt*X

		return Y_new
