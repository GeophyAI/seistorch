# from pytorch_msssim import SSIM as _ssim
from math import exp

import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from torchvision.models import vgg19
# import geomloss
from wavetorch.sinkhorn_pointcloud import sinkhorn_loss
from wavetorch.utils import interp1d
from torch.nn.functional import pad as tpad

class Loss:
    def __init__(self, loss="mse"):
        self.loss_name = loss

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def loss(self, ):
        loss_obj = self._get_loss_object()
        return loss_obj

    def _get_loss_object(self):
        loss_classes = [c for c in globals().values() if isinstance(c, type) and issubclass(c, torch.nn.Module)]
        for loss_class in loss_classes:
            if hasattr(loss_class, "name") and getattr(loss_class(), "name") == self.loss_name:
                return loss_class()
        raise ValueError(f"Cannot find loss named {self.loss_name}")

    
class L1(torch.nn.Module):
    def __init__(self, ):
        super(L1, self).__init__()

    @property
    def name(self,):
        return "l1"
    
    def forward(self, x, y):
        return torch.nn.L1Loss()(x, y)
    
class L2(torch.nn.Module):
    def __init__(self, ):
        super(L2, self).__init__()

    @property
    def name(self,):
        return "l2"
    
    def forward(self, x, y):
        return torch.nn.MSELoss()(x, y)

class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = self.create_window(window_size)

    @property
    def name(self,):
        return "ssim"

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(0)
        _2D_window = _1D_window.t().mm(_1D_window)
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        return window

    def ssim(self, img1, img2, window, window_size, channel):

        # Convert to channel first
        img1 = img1.permute(2, 0, 1)
        img2 = img2.permute(2, 0, 1)

        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1.pow(2), window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2.pow(2), window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = (0.01 * 1.0)**2
        C2 = (0.03 * 1.0)**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # Compute the scaling factors for img1 and img2
        # Normalization
        img1 = img1/(torch.norm(img1, p=2, dim=0)+1e-20)
        img2 = img2/(torch.norm(img2, p=2, dim=0)+1e-20)

        # Compute the scaling factors for img1 and img2
        scale1 = torch.max(torch.abs(img1))
        scale2 = torch.max(torch.abs(img2))
        
        # Normalize img1 and img2 using their respective scaling factors
        img1 = img1 / scale1
        img2 = img2 / scale2

        (_, _, channel) = img1.size()

        if channel == 1 and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size).type_as(img1)
            window = window.expand(channel, 1, self.window_size, self.window_size)
            self.window = window

        return 1 - self.ssim(img1, img2, window, self.window_size, channel)

class L2_Reg(torch.nn.Module):
    def __init__(self, beta=1e-3):
        """
        Initializes the RegularizedLoss

        Args:
            dt (float): The time step. Default is 1.0.
        """
        self.beta = beta
        super().__init__()

    @property
    def name(self,):
        return "l2reg"
    
    def huber_regularization(self, x, delta=1.0):
        """
        Compute the Huber regularization of a tensor.

        Parameters:
        - x: the input tensor.
        - delta: the point at which the loss changes from L1 to L2.

        Returns:
        - The Huber regularization of x.
        """
        abs_x = torch.abs(x)
        quadratic = torch.minimum(abs_x, delta)
        linear = abs_x - quadratic
        return 0.5 * quadratic**2 + delta * linear
    
    def create_LoG_kernel(self, kernel_size=5, sigma=1.4):
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
        kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
        kernel = kernel * (1 - (x ** 2 + y ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / (2 * np.pi * sigma ** 6)
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
        return kernel
    
    def log_regularization(self, image):
        image = image.unsqueeze(0)
        # 3x3 LoG kernel
        log_kernel = self.create_LoG_kernel()# torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float().unsqueeze(0).unsqueeze(0)

        log_kernel = log_kernel.to(image.device)

        # Apply the LoG filter
        edges = F.conv2d(image, log_kernel, padding=1)
        edge_strength = torch.sqrt(edges**2+1e-10)

        # Regularization term is the sum of the edge strength
        regularization = torch.sum(edge_strength)

        return regularization
    
    def edge_preserving(self, image):
        image = image.unsqueeze(0)
        # Sobel filter kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)

        sobel_x = sobel_x.to(image.device)
        sobel_y = sobel_y.to(image.device)

        # Apply the filters to get the edge maps
        edge_x = F.conv2d(image, sobel_x, padding=1)
        edge_y = F.conv2d(image, sobel_y, padding=1)
        
        # Calculate the edge strength
        edge_strength = torch.sqrt(edge_x**2 + edge_y**2 + 1e-10)

        # Regularization term is the sum of the edge strength
        regularization = torch.sum(edge_strength)

        return regularization
    
    def laplace_sharpen(self, m):
        # 定义Laplace滤波器
        laplace_filter = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        laplace_filter = laplace_filter.to(m.device)
        # 图像扩展channel维度并应用滤波器
        m = m[None, None, :]
        m_sharpen = F.conv2d(m, laplace_filter, padding=1)
        
        return m_sharpen[0, 0]
    
    def tv(self, img):
        h, w = img.size()
        return (torch.abs(img[:, :-1] - img[:, 1:]).sum() +torch.abs(img[:-1, :] - img[1:, :]).sum())/(h*w)

    def forward(self, x, y, m):
        mse_loss = torch.nn.MSELoss()(x, y)
        reg_loss = self.beta * self.tv(m) #1e-2
        # reg_loss = self.beta * self.edge_preserving(m)
        # reg_loss = self.beta * self.log_regularization(m)
        return mse_loss+reg_loss

class NormalizedCrossCorrelation(torch.nn.Module):
    def __init__(self, dt=1.0):
        """
        Initializes the NormalizedCrossCorrelation module.

        Args:
            dt (float): The time step. Default is 1.0.
        """
        self.dt = dt
        super().__init__()

    @property
    def name(self,):
        return "ncc"

    def forward(self, x, y):
        """
        Calculates the normalized cross-correlation loss between two sequences of vectors.

        Args:
            x (torch.Tensor): The first sequence of vectors.
            y (torch.Tensor): The second sequence of vectors.

        Returns:
            torch.Tensor: The normalized cross-correlation loss between the sequences.
        """

        x = x/(torch.norm(x, p=2, dim=0)+1e-20)
        y = y/(torch.norm(y, p=2, dim=0)+1e-20)
        loss = torch.nn.MSELoss()(x, y)
        return loss

class Cdist(torch.nn.Module):

    def __init__(self, p=2):
        super(Cdist, self).__init__()
        self.p = p

    @property
    def name(self,):
        return "cdist"

    def forward(self, x, y):
        """
        Compute the Wasserstein loss between x and y.

        Args:
            x: input data, tensor of shape (time_samples, num_traces, num_channels)
            y: target data, tensor of shape (time_samples, num_traces, num_channels)

        Returns:
            A tensor representing the Wasserstein loss.
        """
        # Compute the pairwise distances between x and y
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        distances = torch.cdist(x, y, p=self.p)

        # Compute the Wasserstein loss
        loss = torch.mean(distances)

        return loss

class Wasserstein2d(torch.nn.Module):
    def __init__(self, p=2):
        super(Wasserstein2d, self).__init__()
        self.p = p

    @property
    def name(self,):
        return "wd2d"
    
    def wasserstein_distance_1D(self, pred_waveform, obs_waveform):
        """
        This function computes the Wasserstein distance between the predicted waveform and the observed waveform.

        Parameters:
        pred_waveform (torch.Tensor): The predicted waveform tensor. The dimensions should be (nt, channels).
        obs_waveform (torch.Tensor): The observed waveform tensor. The dimensions should be (nt, channels).

        Returns:
        torch.Tensor: The Wasserstein distance.
        """

        # Sort the data
        sorted_pred_waveform, _ = torch.sort(pred_waveform.reshape(-1))
        sorted_obs_waveform, _ = torch.sort(obs_waveform.reshape(-1))

        # Compute the Wasserstein distance
        wasserstein_dist = torch.abs(sorted_pred_waveform - sorted_obs_waveform).mean()

        return wasserstein_dist

    def forward(self, pred_waveforms, obs_waveforms):
        """
        This function computes the Wasserstein distance for each trace in the batch.

        Parameters:
        pred_waveforms (torch.Tensor): The batch of predicted waveforms. The dimensions should be (nt, ntraces, channels).
        obs_waveforms (torch.Tensor): The batch of observed waveforms. The dimensions should be (nt, ntraces, channels).

        Returns:
        torch.Tensor: The Wasserstein distances for each trace and channel. The dimensions are (ntraces, channels).
        """

        nt, ntraces, channels = pred_waveforms.shape

        # Initialize a tensor to store the Wasserstein distances for each trace and channel
        wasserstein_dists = torch.empty(ntraces, channels)

        # Compute the Wasserstein distance for each trace and channel
        for i in range(ntraces):
            for j in range(channels):
                wasserstein_dists[i, j] = self.wasserstein_distance_1D(pred_waveforms[:, i, j], obs_waveforms[:, i, j])

        return wasserstein_dists.mean()

class NIMl1(torch.nn.Module):
    def __init__(self):
        super().__init__()


    @property
    def name(self,):
        return "niml1"

    def forward(self, x, y):
        """
        Calculates the cumulative distribution distance between two distributions.

        Args:
            x (torch.Tensor): The first distribution.
            y (torch.Tensor): The second distribution.

        Returns:
            torch.Tensor: The cumulative distribution distance between the inputs.
        """

        # Calculate the cumulative distributions of x and y along the first dimension

        # For positive purpose
        x = torch.abs(x)
        y = torch.abs(y)

        # Normalize the positive distributions
        x = x/x.sum()
        y = y/y.sum()

        # Returns the cumulative sum of elements of input in the dimension dim.
        # y(i) = x1 + x2 + ... + xi
        x_cum_dist = torch.cumsum(x, dim=0)
        y_cum_dist = torch.cumsum(y, dim=0)

        # Normalize the cumulative distributions
        # x_cum_dist = x_cum_dist/x_cum_dist.sum()
        # y_cum_dist = y_cum_dist/y_cum_dist.sum()

        return torch.nn.MSELoss()(x_cum_dist, y_cum_dist)


class NIMl2(torch.nn.Module):
    def __init__(self):
        super().__init__()


    @property
    def name(self,):
        return "niml2"

    def forward(self, x, y):
        """
        Calculates the cumulative distribution distance between two distributions.

        Args:
            x (torch.Tensor): The first distribution.
            y (torch.Tensor): The second distribution.

        Returns:
            torch.Tensor: The cumulative distribution distance between the inputs.
        """

        # For positive purpose
        x = torch.pow(x, 2)
        y = torch.pow(y, 2)

        # Normalize the positive distributions
        x = x/x.sum()
        y = y/y.sum()

        # Calculate the cumulative distributions of x and y along the first dimension
        x_cum_dist = torch.cumsum(x, dim=0)
        y_cum_dist = torch.cumsum(y, dim=0)

        # Take the square root of the sum and calculate the mean
        loss = torch.nn.MSELoss()(x_cum_dist, y_cum_dist)

        return loss

class NIMENVELOPE(torch.nn.Module):
    def __init__(self):
        super().__init__()


    @property
    def name(self,):
        return "nimenvelope"
    
    # @torch.no_grad()
    def hilbert(self, data):
        """
        Compute the Hilbert transform of the input data tensor.

        Args:
            data (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The Hilbert transform of the input data tensor.
        """

        nt, _, _ = data.shape
        nfft = 2 ** (nt - 1).bit_length()

        # Compute the FFT
        data_fft = torch.fft.fft(data, n=nfft, dim=0)

        # Create the filter
        h = torch.zeros(nfft, device=data.device).unsqueeze(1).unsqueeze(2)
        # h[0] = 1
        h[1:(nfft // 2)] = 2
        if nfft % 2 == 0:
            h[nfft // 2] = 1

        # Apply the filter and compute the inverse FFT
        hilbert_data_fft = data_fft * h
        hilbert_data = torch.fft.ifft(hilbert_data_fft, dim=0)

        # Truncate the result to the original length
        hilbert_data = hilbert_data[:nt]

        return hilbert_data
    
    def envelope(self, seismograms):
        """
        Compute the envelope of the input seismograms tensor.

        Args:
            seismograms (torch.Tensor): The input seismograms tensor.

        Returns:
            torch.Tensor: The envelope of the input seismograms tensor.
        """
        # Compute the Hilbert transform along the time axis
        hilbert_transform = self.hilbert(seismograms)

        envelope = torch.abs(hilbert_transform)

        return envelope

    def forward(self, x, y):
        """
        Calculates the cumulative distribution distance between two distributions.

        Args:
            x (torch.Tensor): The first distribution.
            y (torch.Tensor): The second distribution.

        Returns:
            torch.Tensor: The cumulative distribution distance between the inputs.
        """

        # For positive purpose
        x = self.envelope(x)
        y = self.envelope(y)

        # Normalize the positive distributions
        x = x/x.sum()
        y = y/y.sum()

        # Calculate the cumulative distributions of x and y along the first dimension
        x_cum_dist = torch.cumsum(x, dim=0)
        y_cum_dist = torch.cumsum(y, dim=0)

        # Take the square root of the sum and calculate the mean
        loss = torch.nn.MSELoss()(x_cum_dist, y_cum_dist)

        return loss

class Phase(torch.nn.Module):
    def __init__(self):
        super(Phase, self).__init__()


    @property
    def name(self,):
        return "phase"


    def phase_correlation(self, obs, sim):
        """
        obs: observed data, tensor of shape (time_samples, num_traces, num_channels)
        sim: simulated data, tensor of shape (time_samples, num_traces, num_channels)
        """
        obs_fft = torch.fft.fft(obs, dim=0)
        sim_fft = torch.fft.fft(sim, dim=0)

        phase_diff = torch.pow(torch.angle(obs_fft) - torch.angle(sim_fft), 2)
        loss = phase_diff.mean()
        return loss

    def forward(self, x, y):
        # pred_phase = self.phase_correlation(x)
        # obs_phase = self.instantaneous_phase(y)

        # loss = F.mse_loss(pred_phase, obs_phase)
        loss = self.phase_correlation(x, y)
        return loss

class Envelope(torch.nn.Module):
    """
    A custom PyTorch module that computes the envelope-based mean squared error loss between
    two input tensors (e.g., predicted and observed seismograms).
    """

    def __init__(self):
        """
        Initialize the parent class.
        """
        super(Envelope, self).__init__()

    @property
    def name(self,):
        return "envelope"

    # @torch.no_grad()
    def hilbert(self, data):
        """
        Compute the Hilbert transform of the input data tensor.

        Args:
            data (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The Hilbert transform of the input data tensor.
        """

        nt, _, _ = data.shape
        nfft = 2 ** (nt - 1).bit_length()

        # Compute the FFT
        data_fft = torch.fft.fft(data, n=nfft, dim=0)

        # Create the filter
        h = torch.zeros(nfft, device=data.device).unsqueeze(1).unsqueeze(2)
        # h[0] = 1
        h[1:(nfft // 2)] = 2
        if nfft % 2 == 0:
            h[nfft // 2] = 1

        # Apply the filter and compute the inverse FFT
        hilbert_data_fft = data_fft * h
        hilbert_data = torch.fft.ifft(hilbert_data_fft, dim=0)

        # Truncate the result to the original length
        hilbert_data = hilbert_data[:nt]

        return hilbert_data
    
    def envelope(self, seismograms):
        """
        Compute the envelope of the input seismograms tensor.

        Args:
            seismograms (torch.Tensor): The input seismograms tensor.

        Returns:
            torch.Tensor: The envelope of the input seismograms tensor.
        """
        # Compute the Hilbert transform along the time axis
        hilbert_transform = self.hilbert(seismograms)

        # Compute the envelope
        # ??
        # analytic_signal = seismograms + \
        #     hilbert_transform.to(seismograms.device) * 1j
        # envelope = torch.abs(analytic_signal)

        # envelope = torch.sqrt(seismograms**2+torch.abs(hilbert_transform)**2)

        envelope = torch.abs(hilbert_transform)

        return envelope

    def envelope_loss(self, pred_seismograms, obs_seismograms):
        """
        Compute the envelope-based mean squared error loss between pred_seismograms and obs_seismograms.

        Args:
            pred_seismograms (torch.Tensor): The predicted seismograms tensor.
            obs_seismograms (torch.Tensor): The observed seismograms tensor.

        Returns:
            torch.Tensor: The computed envelope-based mean squared error loss.
        """
        
        pred_envelope = self.envelope(pred_seismograms)
        obs_envelope = self.envelope(obs_seismograms)

        # pred_envelope = pred_envelope/torch.norm(pred_envelope, p=2, dim=0)
        # obs_envelope = obs_envelope/torch.norm(obs_envelope, p=2, dim=0)

        loss = F.mse_loss(pred_envelope, obs_envelope, reduction="mean")
        return loss

    def forward(self, x, y):
        """
        Compute the envelope-based mean squared error loss for the given input tensors x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed envelope-based mean squared error loss.
        """
        loss = self.envelope_loss(x, y)
        return loss

class Correlation(torch.nn.Module):
    def __init__(self):
        super(Correlation, self).__init__()

    @property
    def name(self,):
        return "cl"

    def forward(self, x, y):

        # Permute from [nt, ntraces, channels] to [nt, channels, ntraces]
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        # Calculate correlation
        correlation = torch.einsum('ijk,ilk->ijl', x, y)

        return -torch.mean(correlation)  # Minimizing -correlation maximizes correlation

class CrossCorrelation(torch.nn.Module):
    """
    A custom PyTorch module that computes a cross-correlation-based loss between 
    two input tensors (e.g., predicted and observed seismograms).
    """

    def __init__(self):
        """
        Initialize the parent class.
        """
        super(CrossCorrelation, self).__init__()

    @property
    def name(self,):
        return "cc"

    def cross_correlation_loss(self, x, y):
        """
        Compute the cross-correlation loss between the input tensors pred_seismograms and obs_seismograms.

        Args:
            x (torch.Tensor): The predicted seismograms tensor.
            y (torch.Tensor): The observed seismograms tensor.

        Returns:
            torch.Tensor: The computed cross-correlation loss.
        """

        # Permute from [nt, ntraces, channels, ] to [ntraces, channels, nt]
        x = x.permute(1, 2, 0)
        y = y.permute(1, 2, 0)

        # Calculate cross corelation
        cross_corr = torch.nn.functional.conv1d(x, y.flip(-1))

        # Calculate the average loss
        loss = torch.mean(cross_corr)

        return loss

    def forward(self, x, y):
        """
        Compute the cross-correlation loss for the given input tensors x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed cross-correlation loss.
        """
        loss = self.cross_correlation_loss(x, y)
        return loss

class Huber(torch.nn.Module):
    def __init__(self, delta=1.0):
        super(Huber, self).__init__()
        self.delta = delta


    @property
    def name(self,):
        return "huber"

    def forward(self, x, y):
        loss = F.smooth_l1_loss(x, y, reduction='mean', beta=self.delta)
        return loss
    
class Traveltime(torch.nn.Module):

    def __init__(self):
        super(Traveltime, self).__init__() 


    @property
    def name(self,):
        return "traveltime"

    def forward(self, x, y):
        x_traveltime = torch.argmax(torch.real(torch.fft.ifft(torch.fft.fft(x**2, dim=0), dim=0)), dim=0)
        y_traveltime = torch.argmax(torch.real(torch.fft.ifft(torch.fft.fft(y**2, dim=0), dim=0)), dim=0)
        loss= torch.nn.MSELoss()(x_traveltime.float(), y_traveltime.float())
        return loss
    
class DTW(torch.nn.Module):
    def __init__(self, window=100):
        super(DTW, self).__init__()
        self.window = window

    @property
    def name(self,):
        return "dtw"

    def forward(self, x, y):
        """
        Compute the Dynamic Time Warping (DTW) loss between x and y.

        Args:
            x: input data, tensor of shape (time_samples, num_traces, num_channels)
            y: target data, tensor of shape (time_samples, num_traces, num_channels)

        Returns:
            A tensor representing the DTW loss.
        """
        # Compute the pairwise distances between x and y for each trace
        loss = 0
        for i in range(x.shape[1]):
            xi = x[:, i, :]
            yi = y[:, i, :]
            distances = torch.cdist(xi, yi, p=2)  # Euclidean distance

            # Compute the accumulated errors using dynamic programming with window constraint
            t, s = xi.shape[0], yi.shape[0]
            acc_error = torch.zeros((t, s))

            # Initialize the first row and first column
            acc_error[0, :] = distances[0, :]
            acc_error[:, 0] = distances[:, 0]

            # Propagate the errors within the window constraint
            for i in range(1, t):
                for j in range(max(1, i - self.window), min(s, i + self.window)):
                    candidates = torch.stack([
                        acc_error[i - 1, j - 1],
                        acc_error[i - 1, j],
                        acc_error[i, j - 1]
                    ])
                    acc_error[i, j] = distances[i, j] + torch.min(candidates)

            # Compute the DTW loss for this trace
            loss += acc_error[-1, -1]

        # Average the loss over all traces
        loss /= x.shape[1]

        return loss

class Sinkhorn(torch.nn.Module):
    r"""
    git@github.com:dfdazac/wassdistance.git
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps=1e-16, max_iter=50, reduction='mean'):
        super(Sinkhorn, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    @property
    def name(self,):
        return "sinkhorn"

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float, device=C.device,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float, device=C.device,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

class Integration(torch.nn.Module):

    def __init__(self, ):
        super(Integration, self).__init__()

    @property
    def name(self,):
        return "integration"
    
    def forward(self, x, y):
        x = torch.abs(x)
        y = torch.abs(y)
        inter_x = torch.cumulative_trapezoid(x, dim=0)
        inter_y = torch.cumulative_trapezoid(y, dim=0)
        return torch.nn.MSELoss()(inter_x, inter_y)

class Test(torch.nn.Module):

    def __init__(self, ):
        super(Test, self).__init__()

    @property
    def name(self,):
        return "test"
    
    def calculate_W2_squared(self, f, g, f_ori=0):
        T0 = f.size(0)
        t = torch.linspace(0, T0, len(f))
        F_t = torch.trapz(f, t)
        
        # Perform interpolation manually
        indices = (F_t * (len(g) - 1) / T0).long()
        G_inv_F_t = torch.gather(g, 0, indices)
        
        integrand = (t - G_inv_F_t) ** 2 * f
        result = torch.trapz(integrand, t)
        return result.item()
    
    def pos(self, d):
        return torch.clamp(d, min=0)
    
    def neg(self, d):
        return torch.clamp(-d, min=0)

    def scale(self, d):
        return d/d.sum()
    
    def calculate_W2(f, g, dt=1.0):
        from scipy import interpolate
        # 计算F和G
        F = torch.cumsum(f, dim=0) * dt
        G = torch.cumsum(g, dim=0) * dt

        # 生成G的逆函数
        t = torch.arange(f.shape[0], device=f.devcie) * dt
        G_inverse = interpolate.interp1d(G.cpu().numpy(), t.cpu().numpy(), fill_value="extrapolate")

        # 计算Wasserstein距离
        W2 = torch.sum((t - torch.tensor(G_inverse(F.detach().cpu().numpy()), dtype=torch.float32).to(f.device))**2 * f) * dt

        return W2
    
    def _wd1d2(self, f, g):
        f = f.squeeze()
        g = g.squeeze()
        # Positive

        f = f**2
        g = g**2

        f = f/f.sum()
        g = g/g.sum()

        F = torch.cumsum(f, 0)
        G = torch.cumsum(g, 0)

        # Method 1
        # t = torch.arange(0, f.size(0), dtype=f.dtype, device=f.device)
        # G_inverse_F = self.linear_interpolation(F, G, t)#.squeeze()
        # difft = t - G_inverse_F
        # loss = torch.sum(difft**2 * f)
        loss = ot.wasserstein_1d(F, G, p=2)

        return loss
    
    def _wd1d(self, f, g):
        f = f.squeeze()
        g = g.squeeze()
        # Positive
        f_pos = self.scale(self.pos(f))
        f_neg = self.scale(self.neg(f))
        g_pos = self.scale(self.pos(g))
        g_neg = self.scale(self.neg(g))

        F_pos = torch.cumsum(f_pos, 0)
        F_neg = torch.cumsum(f_neg, 0)
        G_pos = torch.cumsum(g_pos, 0)
        G_neg = torch.cumsum(g_neg, 0)

        # Method 1
        t = torch.arange(0, f.size(0), dtype=f.dtype, device=f.device)
        G_inverse_F_pos = self.linear_interpolation(F_pos, G_pos, t)#.squeeze()
        G_inverse_F_neg = self.linear_interpolation(F_neg, G_neg, t)#.squeeze()

        difft_pos = t - G_inverse_F_pos
        difft_neg = t - G_inverse_F_neg
        loss_pos = torch.sum(difft_pos**2 * f_pos)
        loss_neg = torch.sum(difft_neg**2 * f_neg)

        # Method 2
        # loss_pos = ot.wasserstein_1d(F_pos.squeeze(), G_pos.squeeze(), p=2)
        # loss_neg = ot.wasserstein_1d(F_neg.squeeze(), G_neg.squeeze(), p=2)

        return loss_pos+loss_neg
    
    def forward(self, x, y):

        loss = 0

        nsamples, ntraces, _ = x.size()
        loss = 0
        for trace in range(ntraces):
            _x, _y = x[:,trace, :], y[:,trace, :]
            loss += self._wd1d(_x, _y)
        return loss
    
    def linear_interpolation(self, t, t_values, y_values):
        """
        Perform linear interpolation. Find the values of y corresponding to t using the given (t_values, y_values) pairs.
        """

        # Make sure the tensor dimensions are compatible
        t = t.unsqueeze(-1)

        # Find the indices of the t_values that are just smaller than t
        indices = torch.searchsorted(t_values, t) - 1
        indices = indices.clamp(min=0, max=t_values.shape[0] - 2)

        # Compute the fraction by which t is larger than the found t_values
        fraction = (t - t_values[indices]) / (t_values[indices + 1] - t_values[indices] + 1e-10)

        # Use the fraction to interpolate between the y_values
        interpolated_values = y_values[indices] + fraction * (y_values[indices + 1] - y_values[indices])

        return interpolated_values.squeeze(-1)

class BatchedSinkhorn(Sinkhorn):
    def __init__(self, batch_size=10, *args, **kwargs):
        super(BatchedSinkhorn, self).__init__(*args, **kwargs)
        self.batch_size = batch_size

    @property
    def name(self,):
        return "bsinkhorn"

    def forward(self, x, y):
        batch_size = x.shape[1]
        total_loss = 0

        # Handle the case where batch_size is less than self.batch_size.
        num_batches = (batch_size + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, batch_size)
            batch_x = x[:, start:end, :]
            batch_y = y[:, start:end, :]
            loss = super().forward(batch_x, batch_y)
            total_loss += loss

        return total_loss / num_batches

class CosineSimilarity(torch.nn.Module):

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
        # Reshape x and y to (time_samples, num_traces * num_channels)
        x_reshaped = x.view(x.shape[0], -1)
        y_reshaped = y.view(y.shape[0], -1)
        # Compute cosine similarity along the second dimension
        similarity = F.cosine_similarity(x_reshaped, y_reshaped, dim=1, eps=1e-10)

        # Compute the mean difference between similarity and 1
        loss = torch.mean(1 - similarity)

        return loss
    
class Convolution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "conv"

    def forward(self, x, y):
        # Reshape the inputs to (batch, channel, height, width)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        
        # Compute cross-correlation using convolution
        correlation = torch.nn.functional.conv2d(x, y.flip(-1).flip(-2))
        # Negative correlation will result in the loss being minimized when the inputs are most similar
        return -correlation

class Chebyshev(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "chebyshev"

    def forward(self, x, y):
        return torch.max(torch.abs(x - y))

class Minkowski(torch.nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    @property
    def name(self,):
        return "minkowski"

    def forward(self, x, y):
        return torch.sum(torch.abs(x - y) ** self.p) ** (1. / self.p)

class PearsonCorrelation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "pcr"

    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        loss = 1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return loss

class KLDivergenceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "kld"

    def forward(self, x, y):
        P = torch.nn.functional.softmax(x, dim=0)
        Q = torch.nn.functional.softmax(y, dim=0)
        return torch.sum(P * torch.log(P / Q))

class Perceptual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(vgg.features)[:18]).cuda()
        self.feature_extractor.eval()
        self.mse_loss = torch.nn.MSELoss()

        # Define the downsampling layer
        self.pool = torch.nn.AvgPool2d(kernel_size=(4, 4), stride=(2, 2))

    @property
    def name(self,):
        return "perceptual"

    def to_rgb(self, x):
        x_mono = x.mean(dim=-1, keepdim=True)  # Take the mean along the channel dimension to get a single-channel image
        x_rgb = x_mono.repeat(1, 1, 3)  # Repeat the single-channel image along the channel dimension to get a three-channel image
        x_rgb = x_rgb.permute(2, 0, 1).unsqueeze(0)  # Change to the format (batch_size, channels, height, width)

        # Apply the pooling operation to downsample the data
        x_rgb = self.pool(x_rgb)

        return x_rgb

    def forward(self, x, y):
        x_rgb = self.to_rgb(x)
        y_rgb = self.to_rgb(y)

        x_features = self.feature_extractor(x_rgb)
        y_features = self.feature_extractor(y_rgb)
        return self.mse_loss(x_features, y_features)

class TotalVariation2D(torch.nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TotalVariation2D, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    @property
    def name(self,):
        return "tv2d"

    def forward(self, x):
        h_x = x.size(0)
        w_x = x.size(1)
        h_tv = torch.pow((x[1:,:]-x[:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,1:]-x[:,:w_x-1]),2).sum()
        return self.tv_loss_weight*2*(h_tv+w_tv)/(h_x*w_x)

class FWILoss(torch.nn.Module):
    def __init__(self, tv_loss_weight=1e-7):
        super(FWILoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight
        self.mse_loss = torch.nn.MSELoss()
        self.tv_loss = TotalVariation2D(tv_loss_weight)

    @property
    def name(self,):
        return "l2+tv2d"

    def forward(self, syn, obs, model):
        loss_fwi = self.mse_loss(syn, obs)
        loss_tv = self.tv_loss(model)
        loss_total = loss_fwi + loss_tv
        return loss_total
