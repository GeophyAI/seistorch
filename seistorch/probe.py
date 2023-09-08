import torch

from .utils import to_tensor


class WaveProbe(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		# Need to be int64
		self.register_buffer('x', to_tensor(x, dtype=torch.int64))
		self.register_buffer('y', to_tensor(y, dtype=torch.int64))

	def __repr__(self,):
		return super().__repr__() + '\nProbe location: x:{} z:{}'.format(self.y, self.x)

	def forward(self, x):
		return x[:, self.x, self.y]

	def plot(self, ax, color='k'):
		marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', markeredgecolor=color, markerfacecolor='none', markeredgewidth=1.0, markersize=4)
		return marker


class WaveIntensityProbe(WaveProbe):
	def __init__(self, x, y):
		#print('WaveIntensityProbe', x, y)
		super().__init__(x, y)

	def forward(self, x):
		return super().forward(x)#.pow(2)
