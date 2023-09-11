#import skimage
import torch

from .utils import to_tensor

class WaveSource(torch.nn.Module):
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

	def coords(self,):
		"""Return the coordinates of the source.

		Returns:
			dict: A list of coordinates.
			Example: {'x': [x1, x2, ..., xn], 
					  'y': [y1, y2, ..., yn], 
					  'z': [z1, z2, ..., zn]]}
		"""
		return dict(zip(self.coord_labels, [getattr(self, key) for key in self.coord_labels]))
	# def forward(self, Y, X, dt=1.0):
	# 	Y_new = Y.clone()
	# 	Y_new[..., self.x, self.y] += dt*X
	# 	return Y_new
	
	# The following implementation have memory leakage problem
	# But I donot know why when I replace the following codes with aboving
	# no-leakage implementation, the elastic fwi goes wrong.

	def get_forward_func(self, ):
		return getattr(self, f"forward{self.ndim}d")

	def forward2d(self, Y, X, dt=1.0):
		#Y_new = Y.clone()
		Y[..., self.y, self.x] += dt*X
		return Y
	
	def forward3d(self, Y, X, dt=1.0):
		#Y_new = Y.clone()
		Y[..., self.x, self.z, self.y] += dt*X
		return Y