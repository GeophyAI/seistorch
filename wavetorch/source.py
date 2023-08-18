#import skimage
import torch

from .utils import to_tensor

class WaveSource(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		# These need to be longs for advanced indexing to work
		self.register_buffer('x', to_tensor(x, dtype=torch.int64))
		self.register_buffer('y', to_tensor(y, dtype=torch.int64))

	def __repr__(self,):
		return super().__repr__() + '\nSource location: x:{} z:{}'.format(self.y, self.x)

	def forward(self, Y, X, dt=1.0):
		Y_new = Y.clone()
		Y_new[..., self.x, self.y] += dt*X
		return Y_new
	
	# The following implementation have memory leakage problem
	# But I donot know why when I replace the following codes with aboving
	# no-leakage implementation, the elastic fwi goes wrong.
	# def forward(self, Y, X, dt=1.0):
	# 	#Y_new = Y.clone()
	# 	Y[..., self.x, self.y] += dt*X
	# 	return Y

	def plot(self, ax, color='r'):
		marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', markeredgecolor=color, markerfacecolor='none', markeredgewidth=1.0, markersize=4)
		return marker
