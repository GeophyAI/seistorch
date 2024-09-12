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
from seistorch.signal import instantaneous_phase as ip
from seistorch.array import SeisArray

from numba import cuda
from numba import jit, prange
import torch.cuda
import math
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

    def loss(self, cfg, *args, **kwargs):
        loss_obj = self._get_loss_object(**kwargs)
        loss_obj.cfg = cfg

        if isinstance(loss_obj, torch.autograd.Function):
            loss_obj = loss_obj.apply

        return loss_obj

    def _get_loss_object(self, **kwargs):
        loss_classes = [c for c in globals().values() if isinstance(c, type) and issubclass(c, torch.nn.Module)]
        loss_classes.extend([c for c in globals().values() if isinstance(c, type) and issubclass(c, torch.autograd.Function)])
        for loss_class in loss_classes:
            if hasattr(loss_class, "name") and getattr(loss_class(), "name") == self.loss_name:
                return loss_class(**kwargs)
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

class EnvelopeCosineSimilarity(torch.nn.Module):
    """The cosine similarity (Normalized cross correlation) loss function.
    """

    def __init__(self):
        super(EnvelopeCosineSimilarity, self).__init__()

    @property
    def name(self,):
        return "ecs"

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
            
            _x = envelope(_x)
            _y = envelope(_y)

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

    def __init__(self, method='subtract'):
        """
        Initialize the parent class.
        """
        super(Envelope, self).__init__()
        self.method = method
        self.env = lambda x: SeisArray(x).envelope()
        self.loss = {'subtract': lambda x, y: ( (x-y)**2 ).sum(), # eq.5
                     'square': lambda x, y: ( (x**2-y**2)**2 ).sum(), # eq.6
                     'log': lambda x, y: torch.log(x/y).sum()}# eq.7
        
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
            _x = self.env(x[i])
            _y = self.env(y[i])
            loss += (0.5*inner_method(_x, _y))
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
        # Compute the instantaneous phase
        hilbert_d = hilbert(d)
        imag, real = hilbert_d.imag, hilbert_d.real
        instantaneous_phase = torch.arctan2(imag, real)
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
        env_x = torch.abs(hilbert(x))
        factor1 = env_x**2
        phi_x = self.instantaneous_phase(x*factor1.detach())
        phi_y = self.instantaneous_phase(y*factor1.detach())

        # Compute the phase difference
        phase_difference = phi_x - phi_y

        return phase_difference
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            loss += torch.sum(0.5*self.instantaneous_phase_diff(_x, _y)**2)
        return loss

class __InstantaneousPhase__(torch.autograd.Function):
    """Yuan et. al, <The exponentiated phase measurement, 
    and objective-function hybridization for adjoint waveform tomography>
    10.1093/gji/ggaa063
    Seisflow implementation: https://github.com/adjtomo/seisflows/blob/master/seisflows/plugins/preprocess/adjoint.py
    """
    @staticmethod
    def forward(ctx, x, y):
        loss = 0.
        ctx.save_for_backward(x, y)
        for _x, _y in zip(x, y):
            env_x = torch.abs(hilbert(_x))
            factor1 = env_x**2
            phi_x = ip(_x*factor1.detach())
            phi_y = ip(_y*factor1.detach())
            loss += torch.sum(0.5*(phi_x-phi_y)**2)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        syn, obs = ctx.saved_tensors
        adj = torch.zeros_like(syn)
        for batch in range(syn.shape[0]):
            _syn = syn[batch]
            _obs = obs[batch]

            hilbert_syn = hilbert(_syn)
            r = torch.real(hilbert_syn)
            i = torch.imag(hilbert_syn)
            phi_syn = torch.arctan2(i, r)

            hilbert_obs = hilbert(_obs)
            r = torch.real(hilbert_obs)
            i = torch.imag(hilbert_obs)
            phi_obs = torch.arctan2(i, r)

            phi_rsd = phi_syn - phi_obs

            adj[batch] = phi_rsd*(hilbert_syn.imag)+hilbert(phi_rsd*_syn).imag

        return adj*grad_output, None
    
class InstantaneousPhase2(torch.nn.Module):
    def __init__(self):
        super(InstantaneousPhase2, self).__init__()

    @property
    def name(self,):
        return "ip2"
    
    def forward(self, x, y):
        loss = __InstantaneousPhase__.apply(x, y)
        return loss

class ExpInstantaneousPhase(torch.nn.Module):

    def __init__(self):
        """
        Initialize the parent class.
        """
        super(ExpInstantaneousPhase, self).__init__()

    @property
    def name(self,):
        return "eip"
    
    def forward(self, x, y):
        loss = 0.
        for _x, _y in zip(x, y):
            nsamples, ntraces, nchannels = _x.shape
            for i in range(ntraces):
                for j in range(nchannels):

                    this_x = _x[:, i:i+1, j:j+1]
                    this_y = _y[:, i:i+1, j:j+1]

                    analy_x = hilbert(this_x)
                    analy_y = hilbert(this_y)

                    Ax = torch.abs(analy_x)#+1e-8
                    Ay = torch.abs(analy_y)#+1e-8

                    part1 = (this_x/Ax + this_y/Ay)**2
                    part2 = (analy_x.imag/Ax + analy_y.imag/Ay)**2
                    
                    loss -= torch.mean(part1+part2)

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
        return ((x-y)**2).sum()

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

class Weighted(torch.nn.Module):
    def __init__(self, loss_names=[], weights=[], cfg=None, **kwargs):
        self.weights = weights
        super(Weighted, self).__init__()
        self.loss = []
        for loss_name in loss_names:
            self.loss.append(Loss(loss_name).loss(cfg))

    @property
    def name(self,):
        return "weighted"

    def forward(self, x, y):
        loss = 0.
        for _loss, _weight in zip(self.loss, self.weights):
            loss += _weight*_loss(x, y)
        return loss

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(torch.autograd.Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(torch.autograd.Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda=False, gamma=100.0, normalize=True, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    @property
    def name(self,):
        return "sdtw"

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
            print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
            use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        # original shape: batch_size x seq_len x traces x dims
        # Permute to: batch_size x traces x seq_len x dims
        # flatten to: (batch_size * traces) x seq_len x dims
        X = X.permute(0, 2, 1, 3).reshape(-1, X.shape[1], X.shape[-1])
        Y = Y.permute(0, 2, 1, 3).reshape(-1, Y.shape[1], Y.shape[-1])
        # Downsample
        # X = X[:,::5,:]
        # Y = Y[:,::5,:]
        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return (out_xy - 1 / 2 * (out_xx + out_yy)).mean()
        else:
            D_xy = self.dist_func(X, Y)
            return (func_dtw(D_xy, self.gamma, self.bandwidth)).mean()

# ----------------------------------------------------------------------------------------------------------------------

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