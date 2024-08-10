# from pytorch_msssim import SSIM as _ssim

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from seistorch.signal import differentiable_trvaletime_difference as dtd
from geomloss import SamplesLoss
from seistorch.transform import *
from torchvision.transforms.functional import gaussian_blur
from seistorch.signal import local_coherence as lc
# try:
#     from torchvision.models import vgg19
# except:
#     print("Cannot import torchvision.models.vgg19")

class Loss:
    """A warpper class for loss functions.
    """
    def __init__(self, loss="mse"):
        self.loss_name = loss

    def __repr__(self,):
        return f"Loss(loss={self.loss_name})"

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def loss(self, cfg):
        loss_obj = self._get_loss_object()
        loss_obj.cfg = cfg

        if isinstance(loss_obj, torch.autograd.Function):
            loss_obj = loss_obj.apply

        return loss_obj

    def _get_loss_object(self,):
        loss_classes = [c for c in globals().values() if isinstance(c, type) and issubclass(c, torch.nn.Module)]
        loss_classes.extend([c for c in globals().values() if isinstance(c, type) and issubclass(c, torch.autograd.Function)])

        for loss_class in loss_classes:
            if hasattr(loss_class, "name") and getattr(loss_class(), "name") == self.loss_name:
                return loss_class()
        raise ValueError(f"Cannot find loss named {self.loss_name}")

class CosineSimilarity(torch.nn.Module):
    """The cosine similarity (Normalized cross correlation) loss function.
    """

    def __init__(self):
        super(CosineSimilarity, self).__init__()

    @property
    def name(self,):
        return "cs"

    def forward(self, x, y):
        """
        Compute the similarity loss based on cosine similarity.

        Args:
            x: input data, tensor of shape (time_samples, num_traces, num_channels)
            y: target data, tensor of shape (time_samples, num_traces, num_channels)

        Returns:
            A tensor representing the similarity loss.
        """
        loss = 0.
        for _x, _y in zip(x, y):
            nt = _x.shape[0]
            # Reshape x and y to (time_samples, num_traces * num_channels)
            x_reshaped = _x.view(nt, -1)
            y_reshaped = _y.view(nt, -1)
            # Compute cosine similarity along the ? dimension
            similarity = F.cosine_similarity(x_reshaped, y_reshaped, dim=0, eps=1e-10)
            # Compute the mean difference between similarity and 1
            loss += torch.mean(1-similarity)

        return loss

class Crosscorrelation(torch.nn.Module):
    """The cosine similarity (Normalized cross correlation) loss function.
    """

    def __init__(self):
        super(Crosscorrelation, self).__init__()

    @property
    def name(self,):
        return "cc"

    def forward(self, x, y, win=512, step=1):
        """
        Compute the similarity loss based on cosine similarity.

        Args:
            x: input data, tensor of shape (time_samples, num_traces, num_channels)
            y: target data, tensor of shape (time_samples, num_traces, num_channels)

        Returns:
            A tensor representing the similarity loss.
        """
        loss = 0.
        for _x, _y in zip(x, y):
            
            nt = _x.shape[0]

            # From (nt, ntraces, nchannle)
            # To   (ntraces, nchannels, nt)
            _x = _x.permute(1, 2, 0)
            _y = _y.permute(1, 2, 0)
            # conv 1d: input shape: (batch_size, in_channels, signal_len)
            #           kernel shape: (out_channels, in_channels, kernel_size)
            for trace in range(_x.shape[0]):
                cross_corr = torch.nn.functional.conv1d(_y[trace].unsqueeze(0), 
                                                        _x[trace].unsqueeze(0))
                loss += -cross_corr.mean()
            # calculate the local cosine similarity
            # for i in range(x_reshaped.shape[0]//step):
            #     # use a slice window
            #     start = i*step
            #     end = i*step+win
            #     end = nt if end > nt else end
            #     drange = range(start, end)
            #     # Compute cosine similarity along the ? dimension
            #     cross_corr = 
            #     similarity = F.cosine_similarity(x_reshaped[drange], y_reshaped[drange], dim=0, eps=1e-22)
            #     # Compute the mean difference between similarity and 1
            #     loss += torch.sum(1-similarity)

        return loss

class Envelope(torch.nn.Module):
    """
    A custom PyTorch module that computes the envelope-based mean squared error loss between
    two input tensors (e.g., predicted and observed seismograms).
    Reference:
    [1]: 10.1016/j.jappgeo.2014.07.010
    """

    def __init__(self, method='square'):
        """
        Initialize the parent class.
        """
        super(Envelope, self).__init__()
        self.method = method
        self.loss = {'subtract': lambda x, y: envelope(x)-envelope(y), # eq.5
                     'square': lambda x, y: envelope(x)**2-envelope(y)**2, # eq.6
                     'log': lambda x, y: torch.log(envelope(x)/envelope(y))}# eq.7
        
        # The log method does not work
    @property
    def name(self,):
        return "envelope"

    def forward(self, x, y):
        inner_method = self.loss[self.method]
        """
        Compute the envelope-based mean squared error loss for the given input tensors x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed envelope-based mean squared error loss.
        """
        loss = 0
        for i in range(x.shape[0]):
            loss += 0.5*torch.sum(inner_method(x[i], y[i])**2)
        return loss

class InstantaneousPhase(torch.nn.Module):
    """Yuan et. al, <The exponentiated phase measurement, 
    and objective-function hybridization for adjoint waveform tomography>
    10.1093/gji/ggaa063
    """

    def __init__(self):
        """
        Initialize the parent class.
        """
        super(InstantaneousPhase, self).__init__()

    @property
    def name(self,):
        return "ip"
    
    def instantaneous_phase(self, d):
        """
        Compute the instantaneous phase of the input data tensor.

        Args:
            data (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The instantaneous phase of the input data tensor.
        """
        # Compute the Hilbert transform along the time axis
        hilbert_transform = hilbert(d).real
        
        # Compute the instantaneous phase
        #instantaneous_phase = torch.arctan(hilbert_transform.real/(d+1e-16))
        instantaneous_phase = torch.arctan2(hilbert_transform, d+1e-16)
        return instantaneous_phase
    
    def instantaneous_phase_diff(self, x, y):
        """
        Compute the instantaneous phase difference between x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed instantaneous phase difference.
        """
        phi_x = self.instantaneous_phase(x)
        phi_y = self.instantaneous_phase(y)

        # Compute the phase difference
        phase_difference = phi_x - phi_y

        # Wrap the phase difference to [-pi, pi]
        phase_difference = torch.remainder(phase_difference + torch.pi, 2 * torch.pi) - torch.pi

        return phase_difference
    
    def instantaneous_phase_diff2(self, x, y):

        hx = hilbert(x).real
        hy = hilbert(y).real

        numerator = x*hy-y*hx
        denominator = x*y+hx*hy

        # phase_difference = torch.arctan2(numerator, denominator+1e-16)
        phase_difference = torch.atan(numerator/(denominator+1e-16))
        return phase_difference
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            # 3.5 Instantaneous phase: measurement challenges: Equation 12
            # loss += torch.mean(torch.square(self.instantaneous_phase_diff2(_x, _y)))
            # 3.3 Instantaneous phase: Equation 9~10
            loss += torch.mean(torch.square(self.instantaneous_phase_diff(_x, _y)))
        return loss

class Integration(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def name(self,):
        return "integration"
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            loss += torch.nn.MSELoss()(integrate(_x), integrate(_y))
        return loss

class L1(torch.nn.Module):
    def __init__(self, ):
        super(L1, self).__init__()

    @property
    def name(self,):
        return "l1"
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            loss += torch.nn.L1Loss(reduction='sum')(_x, _y)
        return loss

class SML1(torch.nn.Module):
    def __init__(self, ):
        super(SML1, self).__init__()

    @property
    def name(self,):
        return "sml1"
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            loss += torch.nn.SmoothL1Loss(reduction='sum', beta=0.001)(_x, _y)
        return loss

class L2(torch.nn.Module):
    def __init__(self, ):
        super(L2, self).__init__()

    @property
    def name(self,):
        return "l2"
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            loss += torch.nn.MSELoss(reduction='sum')(_x, _y)
        return loss

class LocalCoherence(torch.nn.Module):
    def __init__(self, wt=101, wx=11, sigma_tau=21.0, sigma_hx=11.0):
        super(LocalCoherence, self).__init__()
        self.wt = wt
        self.wx = wx
        self.sigma_tau = sigma_tau
        self.sigma_hx = sigma_hx
        self.half_window_tau = wt // 2
        self.half_window_hx = wx // 2
        self.gaussian_kernel = self.create_gaussian_kernel(wt, wx, sigma_tau, sigma_hx)
        self.average_kernel = self.create_average_kernel(wt, wx)

    @property
    def name(self,):
        return "lc"

    def create_gaussian_kernel(self, wt, wx, sigma_tau, sigma_hx):
        """Create a 2D Gaussian kernel."""
        xx = torch.arange(wt)-wt//2
        yy = torch.arange(wx)-wx//2
        kernel_T = torch.exp(- (xx**2 / (2*sigma_tau**2)))
        kernel_X = torch.exp(- (yy**2 / (2*sigma_hx**2)))
        kernel_T = kernel_T / kernel_T.sum()
        kernel_X = kernel_X / kernel_X.sum()
        window = kernel_T.unsqueeze(1).mm(kernel_X.t().unsqueeze(0))
        return window.view(1, 1, wt, wx).to(torch.float32)

    def create_average_kernel(self, wt, wx):
        """Create a 2D average kernel."""
        kernel = torch.ones(wt, wx) / (wt * wx)
        return kernel.view(1, 1, wt, wx).to(torch.float32)
    
    def forward(self, x, y):
        
        cs = lc(x, y, wt=self.wt, wx=self.wx, sigma_tau=self.sigma_tau, sigma_hx=self.sigma_hx)

        loss = (1 - cs).mean()
        
        return loss

class NormalizedIntegrationMethod(torch.nn.Module):
    """Donno et.al, doi: 10.3997/2214-4609.20130411
    """
    def __init__(self, criterion='l2', reduction='sum', method='square'):
        super(NormalizedIntegrationMethod, self).__init__()
        self.method = method
        self.loss = {'l1':torch.nn.L1Loss(reduction=reduction), 
                     'l2':torch.nn.MSELoss(reduction=reduction)}[criterion]

    @property
    def name(self,):
        return "nim"
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            # ensure non-negative eq. (1.2)
            _x, _y = both_nonnegative(_x, _y, type=self.method)

            # weights of each trace
            _x = _x/torch.sum(_x, dim=0, keepdim=True)
            _y = _y/torch.sum(_y, dim=0, keepdim=True)

            # intergration # eq. (1.2)
            _x = torch.cumsum(_x, dim=0)
            _y = torch.cumsum(_y, dim=0)

            # ensure equals to one at the end
            # the denominator in equation (1.2) is wrong
            # x = x/wx
            # y = y/wy
            # the denominator should be the maximum value of the trace
            # so that the value at the end is one
            # x = x / x.max() 
            # y = y / y.max()

            # compute the loss
            loss += self.loss(_x, _y)
        return loss
    
class Phase(torch.nn.Module):
    """
    A custom PyTorch module that computes the phase consistency loss between
    two input tensors (e.g., predicted and observed seismograms).
    """

    def __init__(self):
        """
        Initialize the parent class.
        """
        super(Phase, self).__init__()

    @property
    def name(self,):
        return "phase"

    def analytic(self, data):
        """
        Compute the Hilbert transform of the input data tensor.

        Args:
            data (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The Hilbert transform of the input data tensor.
        """

        nt, _, _ = data.shape
        # nfft = 2 ** (nt - 1).bit_length()
        nfft = nt # the scipy implementation uses this

        # Compute the FFT
        data_fft = torch.fft.fft(data, n=nfft, dim=0)

        # Create the filter
        # h = torch.zeros(nfft, device=data.device).unsqueeze(1).unsqueeze(2)
        h = np.zeros(nfft, dtype=np.float32)

        if nfft % 2 == 0:
            h[0] = h[nfft // 2] = 1
            h[1:nfft // 2] = 2
        else:
            h[0] = 1
            h[1:(nfft + 1) // 2] = 2

        h = np.expand_dims(h, 1)
        h = np.expand_dims(h, 2)
        h = torch.from_numpy(h).to(data.device)
        # h = h.requires_grad_(True)
        # Apply the filter and compute the inverse FFT
        hilbert_data = torch.fft.ifft(data_fft * h, dim=0)

        # Truncate the result to the original length
        #hilbert_data = hilbert_data#[:nt]

        return hilbert_data
    
    def phase_consistency(self, syn, obs):

        syn_real, syn_img = syn.real, syn.imag
        obs_real, obs_img = obs.real, obs.imag

        # Calculate the phase of the simulated and observed signals
        syn_phase = torch.atan2(syn_img, syn_real)
        obs_phase = torch.atan2(obs_img, obs_real)

        # Calculate the phase difference
        phase_difference = syn_phase - obs_phase

        # Wrap the phase difference to [-pi, pi]
        phase_difference = torch.remainder(phase_difference + torch.pi, 2 * torch.pi) - torch.pi

        # Calculate the mean squared error
        loss = torch.mean(torch.square(phase_difference))

        return loss
    
    def forward(self, x, y):
        """
        Compute the phase-based mean squared error loss for the given input tensors x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed phase-based mean squared error loss.
        """
        loss = 0
        for i in range(x.shape[0]):
            loss += self.phase_consistency(self.analytic(x[i]), self.analytic(y[i]))
        return loss
   
class ReverseTimeMigration(torch.autograd.Function):
    def __init__(ctx):
        """
        Initialize the parent class.
        """
        super(ReverseTimeMigration, ctx).__init__()

    def __repr__(self) -> str:
        return "Reverse Time Migration"

    @property
    def name(ctx,):
        return "rtm"
    
    @staticmethod
    def forward(ctx, x, y):
        """No loss for RTM.

        Args:
            x (tensor): synthetic seismograms.
            y (tensor): observed seismograms.
        """

        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        ctx.save_for_backward(x, y)

        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        syn, obs = ctx.saved_tensors
        return obs, None

class Sinkhorn(torch.nn.Module):

    """
    A custom PyTorch module that computes the Sinkorn loss between
    two input tensors (e.g., predicted and observed seismograms).
    """

    def __init__(self):
        """
        Initialize the parent class.
        """
        super(Sinkhorn, self).__init__()

    @property
    def name(self,):
        return "sinkhorn"
    
    def forward(self, x, y):
        """
        Compute the Sinkhorn loss for the given input tensors x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed Sinkhorn loss.
        """
        loss = 0

        for _x, _y in zip(x, y):
            _x = _x**2
            _y = _y**2

            wx = torch.sum(_x, dim=0, keepdim=True)[0]
            wy = torch.sum(_y, dim=0, keepdim=True)[0]

            _x = _x / wx
            _y = _y / wy

            for i in range(x.shape[0]):
                loss += SamplesLoss("sinkhorn", p=2, blur=0.00005)(_x, _y).mean()
            
        return loss

class Traveltime(torch.nn.Module):

    def __init__(self):
        super(Traveltime, self).__init__()

    @property
    def name(self,):
        return "traveltime"
    
    def compute_loss(self, x, y):
        """Compute the traveltime difference between two tensors.

        Args:
            x (tensor): The first input tensor with shape (nb, nt, nr, nc).
            y (tensor): The second input tensor with shape (nb, nt, nr, nc).

        Returns:
            tensor: The computed traveltime difference with shape (nb, 1, nr, nc).
        """
        nb, nt, nr, nc = x.shape

        # Normalization
        max_x = torch.max(torch.abs(x), dim=1, keepdim=True).values
        max_y = torch.max(torch.abs(y), dim=1, keepdim=True).values

        x = x / (max_x+1e-16)
        y = y / (max_y+1e-16)

        x = x.permute(0, 2, 3, 1).contiguous()
        y = y.permute(0, 2, 3, 1).contiguous()

        # from (nb, nt, nr, nc) to (nb*nr*nc, nt)
        x = x.view(nb*nr*nc, nt)
        y = y.view(nb*nr*nc, 1, nt)

        # compute the cross correlation
        cc = F.conv1d(x, y, padding=nt-1, groups=nb*nr*nc)
        cc = cc.view(nb, nr, nc, -1)

        # A differentiable argmax
        max_index = dtd(cc, beta=1)
        # Not differentiable
        # max_index = torch.max(cc, dim=3, keepdim=True).indices
        # Not differentiable
        # max_index = torch.argmax(cc, dim=3, keepdim=True)
        traveltime_diff = (max_index-nt+1)
        traveltime_diff = traveltime_diff.view(nb, 1, nr, nc)**2
        # compute the loss
        loss = traveltime_diff.mean()

        return loss, traveltime_diff
        
    def forward(self, x, y):
        loss, _ = self.compute_loss(x, y)
        return loss

class Wasserstein(torch.nn.Module):
    # Source codes refer to ot.lp.solver_1d.wasserstein
    def __init__(self, method='linear'):
        self.method = method
        super(Wasserstein, self).__init__()

    @property
    def name(self,):
        return "wd"
    
    def argsort(self, a, axis=-1):
        sorted, indices = torch.sort(a, dim=axis)
        return indices

    def sort(self, a, axis=-1):
        sorted0, indices = torch.sort(a, dim=axis)
        return sorted0

    def searchsorted(self, a, v, side='left'):
        right = (side != 'left')
        return torch.searchsorted(a, v, right=right)
    
    def take_along_axis(self, arr, indices, axis):
        return torch.gather(arr, axis, indices)

    def quantile_function(self, qs, cws, xs):
        r""" Computes the quantile function of an empirical distribution

        Parameters
        ----------
        qs: array-like, shape (n,)
            Quantiles at which the quantile function is evaluated
        cws: array-like, shape (m, ...)
            cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
        xs: array-like, shape (n, ...)
            locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

        Returns
        -------
        q: array-like, shape (..., n)
            The quantiles of the distribution
        """
        n = xs.shape[0]

        cws = cws.contiguous()
        qs = qs.contiguous()

        idx = self.searchsorted(cws, qs)
        return self.take_along_axis(xs, torch.clip(idx, 0, n - 1), axis=0)

    def zero_pad(self, a, pad_width, value=0):
        from torch.nn.functional import pad
        # pad_width is an array of ndim tuples indicating how many 0 before and after
        # we need to add. We first need to make it compliant with torch syntax, that
        # starts with the last dim, then second last, etc.
        how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
        return pad(a, how_pad, value=value)

    def wasserstein_1d(self, t, u_weights=None, v_weights=None, p=1, require_sort=True):
        r"""
        This code is modified from ot.lp.solver_1d.wasserstein_1d

        Computes the 1 dimensional OT loss [15] between two (batched) empirical
        distributions

        .. math:
            OT_{loss} = \int_0^1 |cdf_u^{-1}(q) - cdf_v^{-1}(q)|^p dq

        It is formally the p-Wasserstein distance raised to the power p.
        We do so in a vectorized way by first building the individual quantile functions then integrating them.

        This function should be preferred to `emd_1d` whenever the backend is
        different to numpy, and when gradients over
        either sample positions or weights are required.

        Parameters
        ----------
        t: array-like, shape (n, ...)
            locations of the first empirical distribution
        t: array-like, shape (m, ...)
            locations of the second empirical distribution
        u_weights: array-like, shape (n, ...), optional
            weights of the first empirical distribution, if None then uniform weights are used
        v_weights: array-like, shape (m, ...), optional
            weights of the second empirical distribution, if None then uniform weights are used
        p: int, optional
            order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
        require_sort: bool, optional
            sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
            the function, default is True

        Returns
        -------
        cost: float/array-like, shape (...)
            the batched EMD

        References
        ----------
        .. [15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

        """

        assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

        # if require_sort:
        #     u_sorter = self.argsort(t, 0)
        #     u_values = self.take_along_axis(u_values, u_sorter, 0)

        #     v_sorter = self.argsort(v_values, 0)
        #     v_values = self.take_along_axis(v_values, v_sorter, 0)

        #     u_weights = self.take_along_axis(u_weights, u_sorter, 0)
        #     v_weights = self.take_along_axis(v_weights, v_sorter, 0)

        u_cumweights = torch.cumsum(u_weights, 0)
        v_cumweights = torch.cumsum(v_weights, 0)

        qs = self.sort(torch.concatenate((u_cumweights, v_cumweights), 0), 0)
        u_quantiles = self.quantile_function(qs, u_cumweights, t)
        v_quantiles = self.quantile_function(qs, v_cumweights, t)
        qs = self.zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
        delta = qs[1:, ...] - qs[:-1, ...]
        diff_quantiles = torch.abs(u_quantiles - v_quantiles)

        if p == 1:
            return torch.sum(delta * diff_quantiles, axis=0)
        return torch.sum(delta * torch.pow(diff_quantiles, p), axis=0)

    def forward(self, x, y):
        """Compute the Wasserstein distance between two tensors.

        Args:
            x (tensor): The first input tensor with shape (nb, nt, nr, nc).
            y (tensor): The second input tensor with shape (nb, nt, nr, nc).

        Returns:
            tensor: The computed Wasserstein distance with shape (nb, 1, nr, nc).
        """
                
        nt, nr, nc = x[0].shape

        # Enforce non-negative
        x, y = both_nonnegative(x, y, type=self.method)

        # Enforce sum to one on the support
        x = norm(x, dim=1, ntype='sumis1')
        y = norm(y, dim=1, ntype='sumis1')

        # Compute the Wasserstein distance
        loss = 0
        dt = 0.001
        t = torch.linspace(0, nt-1, nt, dtype=x[0].dtype).to(x.device)*dt#[None,:]
        # t /= t.max()
        # loss matrix and normalization

        for _x, _y in zip(x, y):

            # works
            # Ensure nt last
            _x = _x.permute(1, 2, 0).view(-1, nt).contiguous()
            _y = _y.permute(1, 2, 0).view(-1, nt).contiguous()

            for i in range(_x.shape[0]):
                if torch.sum(_x[i])==0 or torch.sum(_y[i])==0:
                                loss +=0.
                else:
                    loss += self.wasserstein_1d(t, _x[i], _y[i], p=2, require_sort=True)

        return loss
    
class Wasserstein1d(torch.nn.Module):

    def __init__(self, method='linear'):
        self.method = method
        super(Wasserstein1d, self).__init__()

    @property
    def name(self,):
        return "w1d"
    
    def transform(self, x, y, method='softplus'):
        if method == 'abs':
            return torch.abs(x), torch.abs(y)
        elif method == 'square':
            return x**2, y**2
        elif method == 'sqrt':
            return torch.sqrt(x**2), torch.sqrt(y**2)
        elif method == 'linear':
            min_value = torch.min(x.detach().min(), y.detach().min())
            min_value = min_value if min_value < 0 else 0
            return x - 1.1*min_value, y - 1.1*min_value
        elif method == 'softplus':
            beta = 0.2
            return torch.log(torch.exp(beta*x)+1), torch.log(torch.exp(beta*y)+1)
        elif method == 'envelope':
            return envelope(x), envelope(y)
        elif method == 'exp':
            beta = 1.
            return torch.exp(beta*x), torch.exp(beta*y)
        else:
            raise ValueError('Invalid method')
        
    # def gaussian_kernel(self, x, x_i, bandwidth):
    #     a = torch.sqrt(torch.tensor(2 * np.pi)).to(x_i.device)
    #     b = torch.exp(-0.5 * ((x - x_i) / bandwidth) ** 2)
    #     return (1 / (bandwidth * a)) * b
        
    # def gaussian_kde(self, signal, bandwidth, x_grid):
    #     return self.gaussian_kernel(x_grid, signal, bandwidth)

    def forward(self, x, y):
        loss = 0.
        # Not sure the following is correct
        for _x, _y in zip(x, y):
            # From others
            _x, _y = self.transform(_x, _y, method=self.method)
            # normalize
            _x = _x / (torch.sum(_x, dim=0, keepdim=True)+1e-18)
            _y = _y / (torch.sum(_y, dim=0, keepdim=True)+1e-18)
            # calculate cdf
            cdf_x = torch.cumsum(_x, dim=0)
            cdf_y = torch.cumsum(_y, dim=0)
            abs_diff = torch.abs(cdf_x - cdf_y)**2
            # calculate the loss
            loss += abs_diff.sum()
        return loss
    
# from ot.lp import wasserstein_1d
# class Wasserstein_OT(torch.nn.Module):
#     def __init__(self, method='linear'):
#         self.method = method
#         super(Wasserstein_OT, self).__init__()

#     @property
#     def name(self,):
#         return "wdot"
    
#     def forward(self, x, y):
#         loss = 0.
#         t = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
#         t = t.unsqueeze(1).unsqueeze(2)
#         for _x, _y in zip(x, y):
#             _x, _y = both_nonnegative(_x, _y, self.method)
#             # normalize
#             _x = _x / (torch.sum(_x, dim=0, keepdim=True)+1e-18)
#             _y = _y / (torch.sum(_y, dim=0, keepdim=True)+1e-18)
#             # Not sure the following is correct
#             loss += wasserstein_1d(t, t, _x, _y, p=2)
#         return loss