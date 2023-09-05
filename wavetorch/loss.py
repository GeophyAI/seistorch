# from pytorch_msssim import SSIM as _ssim
import os
from math import exp

import numpy as np
import ot
import torch
import torch.nn.functional as F
# import torchaudio
from scipy.fftpack import fft, fftfreq
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import pad as tpad
from torch.nn.functional import pairwise_distance

from wavetorch.signal import batch_sta_lta, travel_time_diff, pick_first_arrivals
# import geomloss
from wavetorch.utils import interp1d

# from torchvision.models import vgg19



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
        # img1 = img1/(torch.norm(img1, p=2, dim=0)+1e-20)
        # img2 = img2/(torch.norm(img2, p=2, dim=0)+1e-20)

        # # Compute the scaling factors for img1 and img2
        # scale1 = torch.max(torch.abs(img1))
        # scale2 = torch.max(torch.abs(img2))
        
        # # Normalize img1 and img2 using their respective scaling factors
        # img1 = img1 / scale1
        # img2 = img2 / scale2


        img1 = img1 **2
        img2 = img2 **2

        img1 = img1/torch.sum(img1, dim=0)
        img2 = img2/torch.sum(img2, dim=0)

        img1 = torch.cumsum(img1, dim=0)
        img2 = torch.cumsum(img2, dim=0)

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
    
    def autocorrelation(self, d, dt):
        D = torch.fft.fft(d, dim=0)
        D_mag_sq = torch.abs(D)**2
        E = dt * torch.sum(D_mag_sq)/len(D_mag_sq)
        return E+1e-10

    def forward(self, x, y, dt=0.001):
        """
        Calculates the normalized cross-correlation loss between two sequences of vectors.

        Args:
            x (torch.Tensor): The first sequence of vectors.
            y (torch.Tensor): The second sequence of vectors.

        Returns:
            torch.Tensor: The normalized cross-correlation loss between the sequences.
        """
        loss = 0

        for t in range(x.shape[1]):
            for c in range(x.shape[2]):
                _x = x[:, t, c]
                _y = y[:, t, c]#.flip(0)
                if torch.max(torch.abs(_x)) >0:
                    _xnorm = torch.linalg.norm(_x, ord=2)+1e-10
                    _ynorm = torch.linalg.norm(_y, ord=2)+1e-10
                    _xn = _x / _xnorm
                    _yn = _y / _ynorm
                    # Method 1 not work
                    cc = -torch.dot(_xn*dt, _yn)
                    # Method 2 not work
                    #cc = torch.sum(F.conv1d(_xn.unsqueeze(0).unsqueeze(0), _yn.unsqueeze(0).unsqueeze(0)))
                    # Method 3
                    # _xnorm = self.autocorrelation(_x, dt)
                    # _ynorm = self.autocorrelation(_y, dt)
                    # cc = -torch.dot(_x, _y)/(torch.sqrt(_xnorm*_ynorm))

                    loss += cc
                else:
                    loss += 0.
            #loss += self.ncc(x[i], y[i])

        # x = x.transpose(0, 1).transpose(1, 2)
        # y = y.transpose(0, 1).transpose(1, 2)
        # print(x.shape, y.shape)
        # cc = F.conv1d(x, y)
        # print(cc.shape) # ntraces, ntraces, nchannels

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

    def niml1(self, d):
        d = torch.abs(d)
        scale = torch.sum(d, axis=0)
        # Normalize the positive distributions of each trace
        d = d/scale
        # Returns the cumulative sum of elements of input in the dimension dim.
        # y(i) = x1 + x2 + ... + xi
        d = torch.cumsum(d, axis=0)
        return d

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
        x_cum_dist = self.niml1(x)
        y_cum_dist = self.niml1(x)

        return torch.nn.L1Loss()(x_cum_dist, y_cum_dist)

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

class NIMPN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "nimpn"

    def forward(self, x, y):
        """
        Calculates the cumulative distribution distance between two distributions.

        Args:
            x (torch.Tensor): The first distribution.
            y (torch.Tensor): The second distribution.

        Returns:
            torch.Tensor: The cumulative distribution distance between the inputs.
        """

        # Extract the positive parts of x and y
        x_pos = torch.clamp(x, min=0)
        y_pos = torch.clamp(y, min=0)
        # Normalize the positive part
        x_pos_norm = x_pos / torch.sum(x_pos, dim=0)
        y_pos_norm = y_pos / torch.sum(y_pos, dim=0)
        # Calculate the cumulative distributions of x and y along the first dimension
        x_pos_cum = torch.cumsum(x_pos_norm, dim=0)
        y_pos_cum = torch.cumsum(y_pos_norm, dim=0)

        loss_pos = torch.nn.MSELoss()(x_pos_cum, y_pos_cum)

        # Extract the negative parts of x and y
        x_neg = torch.clamp(-x, min=0)
        y_neg = torch.clamp(-y, min=0)
        # Normalize the negative distributions
        x_neg_norm = x_neg / torch.sum(x_neg, dim=0)
        y_neg_norm = y_neg / torch.sum(y_neg, dim=0)
        # Calculate the cumulative distributions of x and y along the first dimension
        x_neg_cum = torch.cumsum(x_neg_norm, dim=0)
        y_neg_cum = torch.cumsum(y_neg_norm, dim=0)

        loss_neg = torch.nn.MSELoss()(x_neg_cum, y_neg_cum)

        return loss_pos + loss_neg
    
class WD1d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "wd1d"
    
    def pos(self, x):
        return torch.clamp(x, min=0)

    def neg(self, x):
        return torch.clamp(-x, min=0)

    def scale(self, x):
        return x / torch.sum(x, dim=0)

    def wd1d_pos(self, x, y):
        # Get the positive and negative parts of x and y
        x_pos = self.pos(x)
        y_pos = self.pos(y)

        x_pos = torch.cumsum(x_pos, dim=0)
        y_pos = torch.cumsum(y_pos, dim=0)

        x_pos = self.scale(x_pos)
        y_pos = self.scale(y_pos)

        C = torch.cdist(x_pos.unsqueeze(1), y_pos.unsqueeze(1)).to(x.device)
        c = C.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(c)
        row_ind, col_ind = torch.from_numpy(row_ind).to(x.device), torch.from_numpy(col_ind).to(x.device)
        emd_loss = C[row_ind, col_ind].sum()

        return emd_loss

    def wd1d_neg(self, x, y):
        # Get the positive and negative parts of x and y
        x_neg = self.neg(x)
        y_neg = self.neg(y)

        x_neg = torch.cumsum(x_neg, dim=0)
        y_neg = torch.cumsum(y_neg, dim=0)

        x_neg = self.scale(x_neg)
        y_neg = self.scale(y_neg)

        C = torch.cdist(x_neg.unsqueeze(1), y_neg.unsqueeze(1)).to(x.device)
        c = C.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(c)
        row_ind, col_ind = torch.from_numpy(row_ind).to(x.device), torch.from_numpy(col_ind).to(x.device)
        emd_loss = C[row_ind, col_ind].sum()

        return emd_loss

    def otloss(self, syn, obs):

        #syn = syn**2
        #obs = obs**2

        mind = torch.min(torch.min(syn), torch.min(obs))

        nt = syn.shape[0]
        t = torch.arange(nt, device=syn.device)

        # Step1. Make sure they are positive
        syn = syn - mind
        obs = obs - mind

        # Step2. Calculate the cumulative distribution
        # cdf_syn = torch.cumsum(syn, dim=0)
        # cdf_obs = torch.cumsum(obs, dim=0)
        cdf_syn = torch.cumulative_trapezoid(syn, dim=0)
        cdf_obs = torch.cumulative_trapezoid(obs, dim=0)

        # Step3. Normalize the distributions for satisfacing the total mass constraint
        syn_norm = cdf_syn / torch.sum(cdf_syn, dim=0, keepdim=True)
        obs_norm = cdf_obs / torch.sum(cdf_obs, dim=0, keepdim=True)

        idx = torch.searchsorted(obs_norm, syn_norm)
        #idx[idx == t.shape[-1]] = -1
        idx = torch.clamp(idx, max = t.shape[-1]-1)
        # idx = torch.clamp(idx, 0, t.shape[-1])
        tidx = t[idx]

        return ((t[1:] - tidx)**2 * syn_norm).sum()

    def forward(self, x, y):
        """
        Calculates the Wasserstein distance between two distributions.

        Args:
            x (torch.Tensor): The first distribution.
            y (torch.Tensor): The second distribution.

        Returns:
            torch.Tensor: The Wasserstein distance between the inputs.
        """
        nt, ntraces, nchannels = x.shape
        loss = 0.
        for _t in range(ntraces):
            for _nc in range(nchannels):
                _x, _y = x[:, _t, _nc], y[:, _t, _nc]
                loss += self.otloss(_x, _y)
                # loss += self.wd1d_pos(_x, _y) + self.wd1d_neg(_x, _y)

        return loss

class NIMSQUARE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "nimsquare"

    def forward(self, x, y):
        """
        Calculates the cumulative distribution distance between two distributions.

        Args:
            x (torch.Tensor): The first distribution.
            y (torch.Tensor): The second distribution.

        Returns:
            torch.Tensor: The cumulative distribution distance between the inputs.
        """

        x_cum_dist = torch.cumsum(x**2, dim=0)
        x_cum_dist = x_cum_dist / torch.sum(x**2, dim=0)
        y_cum_dist = torch.cumsum(y**2, dim=0)
        y_cum_dist = y_cum_dist / torch.sum(y**2, dim=0)

        return torch.nn.MSELoss()(x_cum_dist, y_cum_dist)

class NIMABS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "nimabs"

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
        x_cum_dist = torch.cumsum(x**2, dim=0)
        x_cum_dist = x_cum_dist / torch.sum(x**2)
        y_cum_dist = torch.cumsum(y**2, dim=0)
        y_cum_dist = y_cum_dist / torch.sum(y**2)
        return torch.nn.MSELoss()(x_cum_dist, y_cum_dist)

class NIMl1ORI(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "niml1_ori"

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
        x_cum_dist = torch.cumsum(x, dim=0)
        y_cum_dist = torch.cumsum(y, dim=0)
        return torch.nn.L1Loss()(x_cum_dist, y_cum_dist)

class NIMl2ORI(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self,):
        return "niml2_ori"

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
        x_cum_dist = torch.cumsum(x, dim=0)
        y_cum_dist = torch.cumsum(y, dim=0)
        return torch.nn.MSELoss()(x_cum_dist, y_cum_dist)

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

    def phase_correlation(self, obs, syn):
        """
        obs: observed data, tensor of shape (time_samples, num_traces, num_channels)
        sim: simulated data, tensor of shape (time_samples, num_traces, num_channels)
        """
        # obs_fft = torch.fft.fft(obs, dim=0)
        # sim_fft = torch.fft.fft(syn, dim=0)

        # Calculate the analytic signal of the inputs
        analytic_syn = self.analytic(syn)
        phase_syn = torch.arctan2(torch.imag(analytic_syn), torch.real(analytic_syn))

        analytic_obs = self.analytic(obs)
        phase_obs = torch.arctan2(torch.imag(analytic_obs), torch.real(analytic_obs))

        pi = 3.1415926
        #phase_diff = torch.pow(torch.angle(obs_fft*180/pi) - torch.angle(sim_fft*180/pi), 2)
        loss = torch.nn.MSELoss()(phase_syn, phase_obs)
        #loss = phase_diff.mean()
        return loss
    
    # def spec(self, x, y):
    #     stft_x = torchaudio.transforms.Spectrogram(n_fft=1024)(x).abs()
    #     stft_y = torchaudio.transforms.Spectrogram(n_fft=1024)(y).abs()

    #     return torch.nn.MSELoss()(stft_x, stft_y)

    def forward(self, x, y):
        # pred_phase = self.phase_correlation(x)
        # obs_phase = self.instantaneous_phase(y)

        # loss = F.mse_loss(pred_phase, obs_phase)
        loss = self.phase_correlation(x, y)
        # nt, ntraces, nchannels = x.shape
        # loss = 0.
        # for _t in range(ntraces):
        #     for _nc in range(nchannels):
        #         _x, _y = x[:, _t, _nc], y[:, _t, _nc]
        #         loss += self.spec(_x, _y)
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
    
    def envelope(self, seismograms):
        """
        Compute the envelope of the input seismograms tensor.

        Args:
            seismograms (torch.Tensor): The input seismograms tensor.

        Returns:
            torch.Tensor: The envelope of the input seismograms tensor.
        """
        # Compute the Hilbert transform along the time axis
        hilbert_transform = self.analytic(seismograms)
        
        # Compute the envelope
        envelope = torch.abs(hilbert_transform)

        # envelope = torch.sqrt(hilbert_transform.real**2 + hilbert_transform.imag**2)

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

        # pred_envelope = pred_envelope/(torch.norm(pred_envelope, p=2, dim=0)+1e-16)
        # obs_envelope = obs_envelope/(torch.norm(obs_envelope, p=2, dim=0)+1e-16)

        # loss = F.mse_loss(pred_envelope, obs_envelope, reduction="mean")
        return torch.nn.MSELoss()(pred_envelope, obs_envelope)

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
    
class Traveltime(torch.autograd.Function):

    def __init__(ctx):
        super(Traveltime, ctx).__init__()

    @property
    def name(ctx,):
        return "traveltime"
        
    @staticmethod
    def forward(ctx, x, y):
        dt = 0.001#ctx.cfg['geom']['dt']
        ctx.save_for_backward(x, y)
        loss = 0
        nsamples, ntraces, nchannels = x.shape
        for t in range(ntraces):
            for c in range(nchannels):
                loss += travel_time_diff(x[:, t, c], y[:, t, c], dt)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        syn, obs = ctx.saved_tensors
        adj = torch.zeros_like(obs)
        dt = 0.001#ctx.cfg['geom']['dt']

        adj[1:-1] = (syn[2:]-syn[:-2])/(2.*dt)
        nsamples, ntraces, nchannels = syn.shape
        for t in range(ntraces):
            for c in range(nchannels):
                tt_diff = travel_time_diff(syn[:, t, c], obs[:, t, c], dt)
                norm = torch.sum(adj[:,t,c] * adj[:,t,c]) * dt
                adj[:,t,c] /= (norm+1e-16)
                adj[:,t,c] *= tt_diff

        adj = adj*grad_output

        return adj, None
    
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
    Sinkhorn divergence between two records :math:`x` and :math:`y`:
    """
    def __init__(self, reg=1e-1):
        super(Sinkhorn, self).__init__()
        self.eps = reg

    @property
    def name(self,):
        return "sinkhorn"
    
    def _sinkhorn(self, x, y):
        nt = x.shape[0]
        t = torch.arange(nt, dtype=torch.float, device=x.device)
        M = ot.dist(t.reshape(-1, 1), t.reshape(-1, 1), metric='euclidean')
        #M /= M.max()#*0.1
        # emd = ot.emd2(x, y, M)

        # x = x**2
        # y = y**2

        x = torch.clamp(x, min=0.)
        y = torch.clamp(y, min=0.)

        x = torch.cumsum(x, dim=0)
        y = torch.cumsum(y, dim=0)

        x = x/x.sum()
        y = y/y.sum()

        emd = ot.emd2_1d(x, y, t, t)
        #emd = ot.emd(x, y, M)
        #emd = ot.sinkhorn2(x, y, M, 0.)
        return emd

    def forward(self, x, y):
        nt, ntraces, nchannels = x.shape
        loss = 0.
        for _t in range(ntraces):
            for _nc in range(nchannels):
                _x, _y = x[:, _t, _nc], y[:, _t, _nc]
                loss += self._sinkhorn(_x, _y)
        return loss

class Integration(torch.nn.Module):

    def __init__(self, ):
        super(Integration, self).__init__()

    @property
    def name(self,):
        return "integration"
    
    def intergrate(self, x, times=3):
        for i in range(times):
            x = torch.cumulative_trapezoid(x, dim=0)
        return x
    
    def forward(self, x, y):
        inter_x = self.intergrate(x)
        inter_y = self.intergrate(y)
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

        nsamples, ntraces, channels = x.size()
        loss = 0
        for trace in range(ntraces):
            for c in range(channels):
                _x, _y = x[:,trace, c], y[:,trace, c]
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
        # x_reshaped = x
        # y_reshaped = y
        # print(x_reshaped.shape, y_reshaped.shape)
        # Compute cosine similarity along the ? dimension
        similarity = F.cosine_similarity(x_reshaped, y_reshaped, dim=0, eps=1e-10)
        #print(similarity)
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

        x = x**2
        y = y**2

        x = torch.cumsum(x, dim=0)
        y = torch.cumsum(y, dim=0)

        P = x/torch.sum(x, dim=0)
        Q = y/torch.sum(y, dim=0)

        P = torch.log(P)

        # P = torch.nn.functional.log_softmax(x, dim=0)
        # Q = torch.nn.functional.softmax(y, dim=0)
        return torch.nn.KLDivLoss()(P, Q)

# class Perceptual(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         vgg = vgg19(pretrained=True)
#         self.feature_extractor = torch.nn.Sequential(*list(vgg.features)[:18]).cuda()
#         self.feature_extractor.eval()
#         self.mse_loss = torch.nn.MSELoss()

#         # Define the downsampling layer
#         self.pool = torch.nn.AvgPool2d(kernel_size=(4, 4), stride=(2, 2))

#     @property
#     def name(self,):
#         return "perceptual"

#     def to_rgb(self, x):
#         x_mono = x.mean(dim=-1, keepdim=True)  # Take the mean along the channel dimension to get a single-channel image
#         x_rgb = x_mono.repeat(1, 1, 3)  # Repeat the single-channel image along the channel dimension to get a three-channel image
#         x_rgb = x_rgb.permute(2, 0, 1).unsqueeze(0)  # Change to the format (batch_size, channels, height, width)

#         # Apply the pooling operation to downsample the data
#         x_rgb = self.pool(x_rgb)

#         return x_rgb

#     def forward(self, x, y):
#         x_rgb = self.to_rgb(x)
#         y_rgb = self.to_rgb(y)

#         x_features = self.feature_extractor(x_rgb)
#         y_features = self.feature_extractor(y_rgb)
#         return self.mse_loss(x_features, y_features)

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

class Laplace(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    @property
    def name(self,):
        return "laplace"

    def laplace(self, x, dt=0.001, damp=4):
        #$$u(s)=\int_0^{\infty} u(t) e^{-s t} \mathrm{~d} t,$$
        t = torch.arange(0, x.shape[0], dtype=x.dtype, device=x.device)
        t = t*dt
        return torch.trapz(x*torch.exp(-damp*t[:, None, None]), t, dim=0)

    def forward(self, x, y):
        lap_x = self.laplace(x)
        lap_y = self.laplace(y)
        return torch.log(lap_x / lap_y).mean()

class FATT(torch.autograd.Function):

    @property
    def name(ctx,):
        return "fa"
    
    @staticmethod
    def forward(ctx, x, y):
        """Calculate the first arrival traveltime difference

        Args:
            x (Tensor): Synthetic data.
            y (Tensor): Observered data.
        """
        syn = x
        obs = y
        device = x.device
        """Configures"""
        nsta = 101
        nlta = 1001
        thresh_on = 0.5
        thresh_off = 1.0
        params = {"sta_len": nsta,
                    "lta_len": nlta, 
                    "threshold_on": thresh_on, 
                    "threshold_off": thresh_off}

        nsamples, ntraces, nchannels = x.shape
        rec_idxs = np.arange(0, ntraces)

        # Calculate the first arrival time for each trace
        picked_obs = pick_first_arrivals(obs, **params) # Tensor
        picked_syn = pick_first_arrivals(syn, **params) # Tensor

        # np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa/obs_arr_picked.npy", picked_obs.cpu().detach().numpy())
        # np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa/syn_arr_picked.npy", picked_syn.cpu().detach().numpy())

        fitted_obs = []
        fitted_syn = []

        # Calculate the fitted time of each channel
        for c in range(nchannels):
            # Fit for making the picked first arrival time smooth
            coefficients = np.polyfit(x=rec_idxs, y=picked_obs.cpu().numpy()[..., c], deg=15) # Numpy
            polynomial_obs = np.poly1d(coefficients) # Numpy

            coefficients = np.polyfit(x=rec_idxs, y=picked_syn.cpu().numpy()[..., c], deg=15) # Numpy
            polynomial_syn = np.poly1d(coefficients) # Numpy

            fitted_obs.append(polynomial_obs(rec_idxs))
            fitted_syn.append(polynomial_syn(rec_idxs))

        fitted_obs = torch.from_numpy(np.stack(fitted_obs, axis=-1)).to(device).float() # Tensor
        fitted_syn = torch.from_numpy(np.stack(fitted_syn, axis=-1)).to(device).float() # Tensor

        fitted_obs[fitted_obs<0] =  0
        fitted_syn[fitted_syn<0] =  0
        fitted_obs[fitted_obs>=nsamples] =  nsamples-1
        fitted_syn[fitted_syn>=nsamples] =  nsamples-1

        # np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa/obs_arr_fitted.npy", fitted_obs.cpu().detach().numpy())
        # np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa/syn_arr_fitted.npy", fitted_syn.cpu().detach().numpy())

        ctx.save_for_backward(syn, obs, fitted_syn, fitted_obs)

        return torch.nn.MSELoss()(fitted_syn, fitted_obs)
    
    @staticmethod
    def backward(ctx, grad_output):
        syn, obs, fitted_syn, fitted_obs = ctx.saved_tensors
        adj = torch.zeros_like(obs)
        dt = 0.001#ctx.cfg['geom']['dt']

        window_time = 800 #ms
        windows_nt = int(window_time/1000/dt)

        nsamples, ntraces, nchannels = syn.shape

        # Generate the data mask
        mask_obs = torch.zeros_like(obs)
        mask_syn = torch.zeros_like(syn)
        row_indices = torch.arange(nsamples).unsqueeze(1).to(obs.device)
        for c in range(nchannels):
            # Generate indices of ones for the observed data
            start = torch.maximum(torch.zeros_like(fitted_obs[..., c]), fitted_obs[..., c].int()-100)
            end = torch.minimum(torch.ones_like(fitted_obs[..., c])*nsamples, fitted_obs[..., c].int()+windows_nt-100)
            ones_indices_obs = (row_indices >= start.unsqueeze(0)) & (row_indices <= end.unsqueeze(0))
            # Generate indices of ones for the synthetic data
            start = torch.maximum(torch.zeros_like(fitted_syn[..., c]), fitted_syn[..., c].int()-100)
            end = torch.minimum(torch.ones_like(fitted_syn[..., c])*nsamples, fitted_syn[..., c].int()+windows_nt-100)
            ones_indices_syn = (row_indices >= start.unsqueeze(0)) & (row_indices <= end.unsqueeze(0))
            # Set the mask
            mask_obs[..., c][ones_indices_obs] = 1.
            mask_syn[..., c][ones_indices_syn] = 1.

        # Mask the data so that only the first arrivals are used
        masked_obs = obs * mask_obs
        masked_syn = syn * mask_syn

        # np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa2/mask_obs.npy", mask_obs.cpu().detach().numpy())
        # np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa2/mask_syn.npy", mask_syn.cpu().detach().numpy())

        np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa2/masked_obs.npy", masked_obs.cpu().detach().numpy())
        np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa2/masked_syn.npy", masked_syn.cpu().detach().numpy())

        adj[1:-1] = (masked_syn[2:]-masked_syn[:-2])/(2.*dt)

        # Calculate the adjoint source trace by trace
        for t in range(ntraces):
            for c in range(nchannels):
                tt_diff = -travel_time_diff(masked_obs[:, t, c], masked_syn[:, t, c], dt)
                norm = torch.sum(adj[:,t,c] * adj[:,t,c]) * dt
                adj[:,t,c] /= (norm+1e-16)
                adj[:,t,c] *= tt_diff

        adj = adj*grad_output

        np.save("/public1/home/wangsw/FWI/NO_LOWFREQ/fa2/adj.npy", adj.cpu().detach().numpy())

        return adj, None


class FirstArrivalTravelTime(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def setup(self,):
        self.nsta = 101
        self.nlta = 1001
        self.thresh_on = 0.5
        self.thresh_off = 1.0
        self.dt = self.cfg['geom']['dt']

    @property
    def name(self,):
        return "fatt"
    

    def intersection(self, arrays: list):
        if not arrays:
            return []

        arrays_np = torch.stack(arrays, axis=0)
        intersection_result = torch.any(arrays_np, axis=0)
        
        return intersection_result

    def interpolate(self, x, nsamples):
        device = x.device
        ntraces, nchannels = x.shape
        rec_idxs = np.arange(0, ntraces)

        fitted = []
        # Calculate the fitted time of each channel
        for c in range(nchannels):
            # Fit for making the picked first arrival time smooth
            coefficients = np.polyfit(x=rec_idxs, y=x.cpu().numpy()[..., c], deg=15) # Numpy
            polynomial = np.poly1d(coefficients) # Numpy

            fitted.append(polynomial(rec_idxs))

        fitted = torch.from_numpy(np.stack(fitted, axis=-1)).to(device).float() # Tensor

        fitted[fitted<0] =  0
        fitted[fitted>=nsamples] =  nsamples-1

        return fitted

    def forward(self, x, y):
        # x: Syn
        # y: Obs
        self.setup()
        nsamples, ntraces, nchannles = x.shape
        params = {"sta_len": self.nsta,
                  "lta_len": self.nlta, 
                  "threshold_on": self.thresh_on, 
                  "threshold_off": self.thresh_off}
        # Pick the first arrival travel time
        # x_arrivals = self.process_channels(x, **params)
        # y_arrivals = self.process_channels(y, **params)

        # Calculate the first arrival time for each trace
        x_arrivals = pick_first_arrivals(x, **params) # Tensor
        y_arrivals = pick_first_arrivals(y, **params) # Tensor

        # Interpolate the first arrival time
        x_arrivals = self.interpolate(x_arrivals, nsamples)
        y_arrivals = self.interpolate(y_arrivals, nsamples)

        np.save(f"{self.cfg['ROOTPATH']}/syn_arrivals.npy", x_arrivals.cpu().numpy())
        np.save(f"{self.cfg['ROOTPATH']}/obs_arrivals.npy", y_arrivals.cpu().numpy())
        # Cut the traces according to the arrivals
        # Only retain the events before the first arrival + a fixed time window
        t_after = 600 # ms
        n_reserve = int(t_after/1000/self.dt)

        # Construct the boolean index which is used to select the data
        # Only retain the events before the first arrival + a fixed time window
        # x_index = torch.arange(self.dt, device=x.device)[:, None] < x_arrivals+n_reserve
        # y_index = torch.arange(self.dt, device=x.device)[:, None] < y_arrivals+n_reserve

        x_index = torch.empty_like(x, device=x.device).bool()
        y_index = torch.empty_like(y, device=x.device).bool()
        for i in range(nchannles):
            x_index[...,i] = torch.arange(nsamples, device=x.device)[:, None] < x_arrivals[...,i]+n_reserve
            y_index[...,i] = torch.arange(nsamples, device=x.device)[:, None] < y_arrivals[...,i]+n_reserve

        # Get first arrival
        x_cut = x_index * x
        y_cut = y_index * y

        # x_cut = x
        # y_cut = y

        np.save(f"{self.cfg['ROOTPATH']}/syn.npy", x_cut.cpu().detach().numpy())
        np.save(f"{self.cfg['ROOTPATH']}/obs.npy", y_cut.cpu().detach().numpy())
        
        padding = nsamples - 1
        # Calculate the trace-wise cross-correlation
        loss = 0.
        scale = 1e6
        ttd = np.zeros(ntraces, np.float32)
        argmax = np.zeros(ntraces, np.float32)
        indices = torch.arange(2*nsamples-1, device=x.device)
        for t in range(ntraces):
            for c in range(nchannles):
                _x = x_cut[:, t, c]
                _y = y_cut[:, t, c]

                if torch.max(torch.abs(_x))>1e-5 or torch.max(torch.abs(_y))>1e-5:
                    #cc = torch.abs(F.conv1d(_x.unsqueeze(0), _y.unsqueeze(0).unsqueeze(0), padding=padding))
                    cc = F.conv1d(_x.unsqueeze(0), _y.unsqueeze(0).unsqueeze(0), padding=padding)
                    # using gumbel-softmax for differentiable argmax
                    # in logits, the maximum value is 1, and the others are 0
                    # logits = cc.softmax(dim=-1)
                    logits = F.gumbel_softmax(cc*scale, tau=1, hard=True)
                    max_index = torch.sum(indices * logits)
                    ttd[t] = (max_index.detach().cpu().numpy()-nsamples+1)*self.dt
                    argmax[t] = (torch.argmax(cc).cpu().numpy()-nsamples+1)*self.dt
                    loss += (max_index-nsamples+1)*self.dt
                else:
                    loss += 0.
        np.save(f"{self.cfg['ROOTPATH']}/ttd.npy", ttd)
        np.save(f"{self.cfg['ROOTPATH']}/ttd_argmax.npy", argmax)

        return loss

# class SourceEncoding(torch.nn.Module):
#     def __init__(self,):
#         super().__init__()
    
#     @property
#     def name(self,):
#         return "se"
    
#     def setup(self, nrec, period, dobs):

#         root_path = self.cfg['ROOTPATH']
#         nevents = self.cfg['geom']['Nshots']
#         dt = self.cfg['geom']['dt']
#         bw_l = self.cfg['source_encoding']['BW_L']
#         bw_h = self.cfg['source_encoding']['BW_H']
#         max_freq_shift = self.cfg['source_encoding']['MAX_FREQ_SHIFT']
#         if 'GAMMA' not in self.cfg['source_encoding'].keys():
#             self.cfg['source_encoding']['GAMMA'] = 0
#         gamma = self.cfg['source_encoding']['GAMMA']
#         f0 = self.cfg['source_encoding']['F0']

#         freq_min = float(bw_l)
#         freq_max = float(bw_h)+float(max_freq_shift)

#         #create a mask on relevant frequencies
#         freq  = fftfreq(period, dt)
#         df = freq[1] - freq[0]
#         m = np.array(np.where( (freq_min <= freq) & (freq <= freq_max ) ))
#         num_freqs = m.shape[1]
#         print ('number of frequencies considered : ' +str(num_freqs))
#         print ("")
#         print ('frequency step : ' + str(df) + ' Hz')
#         print ("")
#         stf_file = os.path.join(root_path +'/wavelet.npy')
#         ft_stf_file = os.path.join(root_path +'/ft_stf.npy')
#         if not os.path.isfile(ft_stf_file):
#             tmp=np.load(stf_file)
#             stf_raw=tmp
#             dt_stf = dt
#             nt_stf_tmp = stf_raw.shape[0]
#             ntcal = nt_stf_tmp
#             if nt_stf_tmp>period:
#                 ntcal = period
#             stf=np.zeros(period)
#             stf[:ntcal] = stf_raw[:ntcal,1]
#             #assert period == stf.shape[0]
#             #calculate the laplace coeff of the source time function
#             ft_stf = np.fft.fft(stf*np.exp(-self.gamma*(np.arange(period)*self.dt)))
#             #czt(x=stf[:period,1],m=period,a=self.a)
#             #ft_stf = fft(stf[:period,1])
#             np.save(ft_stf_file,ft_stf[m[0,:]])

#         t_period = 1./f0
#         # calculate early-arrival time for the observed data and save it
#         # for all the source-receiver pairs
#         if not os.path.isfile(os.path.join(self.cfg['ROOTPATH'], 't0array.npy')):
#             self.t0array = np.zeros((self.nrec,nevents))
#             for isrc in range(nevents):
#                 self.t0array[:,isrc] = dt*batch_sta_lta(dobs, dt, t_period)
#             np.save(os.path.join(self.cfg['ROOTPATH'], 't0array.npy'), self.t0array)
#         else:
#             self.t0array = np.load(os.path.join(self.cfg['ROOTPATH'], 't0array.npy'))

#     def forward(self, x, y):
#         loss = 0.
#         self.nsamples, self.nrecs, self.nchannels = x.shape
#         self.setup(self.nrecs, self.nsamples, dobs=y)
#         return loss

import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

# DTW
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
def jacobean_product_squared_euclidean(X, Y, Bt):
    '''
    jacobean_product_squared_euclidean(X, Y, Bt):
    
    Jacobean product of squared Euclidean distance matrix and alignment matrix.
    See equations 2 and 2.5 of https://arxiv.org/abs/1703.01541
    '''
    # print(X.shape, Y.shape, Bt.shape)
    
    ones = torch.ones(Y.shape).to('cuda' if Bt.is_cuda else 'cpu')
    return 2 * (ones.matmul(Bt) * X - Y.matmul(Bt))

class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, X, Y, D, gamma, bandwidth):
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
        ctx.save_for_backward(D, X, Y, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, X, Y, R, gamma, bandwidth = ctx.saved_tensors

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
        G = jacobean_product_squared_euclidean(X.transpose(1,2), Y.transpose(1,2), E.transpose(1,2)).transpose(1,2)

        return grad_output.view(-1, 1, 1).expand_as(G) * G, None, None, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda=True, gamma=0.01, normalize=False, bandwidth=None, dist_func=None):
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

        assert use_cuda, "Only the CUDA version is supported."

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

        if use_cuda and (lx > 4096 or ly > 4096):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 4096 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply

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

    def forward(self, x, y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        nt, nr, nc = x.shape
        loss = 0.
        for c in range(nc):
            for r in range((nr)):
                X = x[::10,r,c].unsqueeze(0).unsqueeze(2)
                Y = y[::10,r,c].unsqueeze(0).unsqueeze(2)

                func_dtw = self._get_func_dtw(X, Y)
                D_xy = self.dist_func(X, Y)
                loss += func_dtw(X, Y, D_xy, self.gamma, self.bandwidth)

        return loss
        # # Check the inputs and get the correct implementation
        # func_dtw = self._get_func_dtw(X, Y)

        # if self.normalize:
        #     # Stack everything up and run
        #     x = torch.cat([X, X, Y])
        #     y = torch.cat([Y, X, Y])
        #     D = self.dist_func(x, y)
        #     out = func_dtw(X, Y, D, self.gamma, self.bandwidth)
        #     out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
        #     return out_xy - 1 / 2 * (out_xx + out_yy)
        # else:
        #     D_xy = self.dist_func(X, Y)
        #     return func_dtw(X, Y, D_xy, self.gamma, self.bandwidth)