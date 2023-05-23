import torch
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
# from pytorch_msssim import SSIM as _ssim
from math import exp
from torchvision.models import vgg19


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
        super(L1, self).__init__()

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
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1.pow(2), window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2.pow(2), window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
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
    def __init__(self, beta=0.001):
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

    def forward(self, x, y, m):
        mse_loss = torch.nn.MSELoss()(x, y)
        reg_loss = self.beta * torch.norm(m.grad, p=2)
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
        loss = 0

        # Iterate over each pair of vectors from x and y
        for _x, _y in zip(x, y):
            # Calculate the L2 norm of each vector along axis 0
            normx = torch.linalg.norm(_x, ord=2, axis=0)
            normy = torch.linalg.norm(_y, ord=2, axis=0)

            # Normalize each vector by dividing it by its norm (adding a small value to avoid division by zero)
            nx = _x / (normx + 1e-12)
            ny = _y / (normy + 1e-12)

            # Calculate the element-wise square of the difference between the normalized vectors and calculate the mean
            loss += torch.mean(torch.pow(nx - ny, 2))

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

class Wasserstein(torch.nn.Module):
    def __init__(self, p=2):
        super(Wasserstein, self).__init__()
        self.p = p

    @property
    def name(self,):
        return "wd"

    def forward(self, x, y):
        """
        Compute the Wasserstein loss between x and y.

        Args:
            x: input data, tensor of shape (time_samples, num_traces, num_channels)
            y: target data, tensor of shape (time_samples, num_traces, num_channels)

        Returns:
            A tensor representing the Wasserstein loss.
        """
        # Compute the pairwise distances between x and y for each trace
        loss = 0
        for i in range(x.shape[1]):
            xi = x[:, i, :].view(x.shape[0], -1)
            yi = y[:, i, :].view(y.shape[0], -1)
            distances = torch.cdist(xi, yi, p=self.p)

            # Compute the Wasserstein loss for this trace
            loss += torch.mean(distances)

        # Average the loss over all traces
        loss /= x.shape[1]

        return loss

class NIM(torch.nn.Module):
    def __init__(self):
        super().__init__()


    @property
    def name(self,):
        return "nim"

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
        x_cum_distributions = torch.cumsum(x, dim=0)
        y_cum_distributions = torch.cumsum(y, dim=0)

        # Calculate the difference between the cumulative distributions
        diff_cum_distributions = x_cum_distributions - y_cum_distributions

        # Calculate the element-wise square of the differences and sum them along the first dimension
        squared_diff_cum_distributions = diff_cum_distributions ** 2
        sum_squared_diff = torch.sum(squared_diff_cum_distributions, dim=0)

        # Take the square root of the sum and calculate the mean
        loss = torch.mean(torch.sqrt(sum_squared_diff))

        return loss

class Phase(torch.nn.Module):
    def __init__(self):
        super(Phase, self).__init__()


    @property
    def name(self,):
        return "phase"

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
        h[0] = 1
        h[1:(nfft // 2)] = 2
        if nfft % 2 == 0:
            h[nfft // 2] = 1

        # Apply the filter and compute the inverse FFT
        hilbert_data_fft = data_fft * h
        hilbert_data = torch.fft.ifft(hilbert_data_fft, dim=0)

        # Truncate the result to the original length
        hilbert_data = hilbert_data[:nt]

        return hilbert_data.real   


    def instantaneous_phase(self, seismograms, mask_value=1e-5):
        # 生成一个mask用于去噪
        mask = torch.ones_like(seismograms).to(seismograms.device)
        mask[torch.abs(seismograms)<mask_value] = 0

        hilbert_transform = self.hilbert(seismograms)

        analytic_signal = seismograms + \
            hilbert_transform.to(seismograms.device) * 1j
        
        phase = torch.angle(analytic_signal)*180/3.14159

        return phase*mask
    
    def phase_correlation(self, obs, sim):
        """
        obs: observed data, tensor of shape (time_samples, num_traces, num_channels)
        sim: simulated data, tensor of shape (time_samples, num_traces, num_channels)
        """
        obs_fft = torch.fft.fft(obs, dim=0)
        sim_fft = torch.fft.fft(sim, dim=0)
        phase_diff = torch.abs(torch.angle(obs_fft) - torch.angle(sim_fft))
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
        h[0] = 1
        h[1:(nfft // 2)] = 2
        if nfft % 2 == 0:
            h[nfft // 2] = 1

        # Apply the filter and compute the inverse FFT
        hilbert_data_fft = data_fft * h
        hilbert_data = torch.fft.ifft(hilbert_data_fft, dim=0)

        # Truncate the result to the original length
        hilbert_data = hilbert_data[:nt]

        return hilbert_data.real

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
        analytic_signal = seismograms + \
            hilbert_transform.to(seismograms.device) * 1j
        envelope = torch.abs(analytic_signal)
        return envelope

    def envelope_mse_loss(self, pred_seismograms, obs_seismograms):
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

        mse_loss = F.mse_loss(pred_envelope, obs_envelope)
        return mse_loss

    def forward(self, x, y):
        """
        Compute the envelope-based mean squared error loss for the given input tensors x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed envelope-based mean squared error loss.
        """
        loss = self.envelope_mse_loss(x, y)
        return loss

class Correlation(torch.nn.Module):
    def __init__(self):
        super(Correlation, self).__init__()

    @property
    def name(self,):
        return "cl"

    def forward(self, pred, target):
        pred_mean = torch.mean(pred, dim=0, keepdim=True)
        target_mean = torch.mean(target, dim=0, keepdim=True)
        pred_std = torch.std(pred, dim=0, keepdim=True)
        target_std = torch.std(target, dim=0, keepdim=True)

        correlation = torch.mean(
            (pred - pred_mean) * (target - target_mean), dim=0) / (pred_std * target_std)

        return -correlation.mean()  # Minimizing -correlation maximizes correlation

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

    def cross_correlation_loss(self, pred_seismograms, obs_seismograms):
        """
        Compute the cross-correlation loss between the input tensors pred_seismograms and obs_seismograms.

        Args:
            pred_seismograms (torch.Tensor): The predicted seismograms tensor.
            obs_seismograms (torch.Tensor): The observed seismograms tensor.

        Returns:
            torch.Tensor: The computed cross-correlation loss.
        """
        # Get the dimensions of the input tensors
        nt, ntraces, _ = pred_seismograms.shape

        # Normalize each trace in the predicted seismograms
        pred_normalized = pred_seismograms / \
            (pred_seismograms.norm(dim=0) + 1e-8)
        # Normalize each trace in the observed seismograms
        obs_normalized = obs_seismograms / (obs_seismograms.norm(dim=0) + 1e-8)

        # Compute the cross-correlation between the normalized predicted and observed seismograms
        cross_correlation = torch.sum(pred_normalized * obs_normalized, dim=0)
        # Calculate the cross-correlation loss as the mean of (1 - cross_correlation)
        cc_loss = 1.0 - torch.mean(cross_correlation)

        return cc_loss

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
    def __init__(self, eps=0.01, max_iter=50):
        super(Sinkhorn, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

    @property
    def name(self,):
        return "sinkhorn"

    def cost_matrix(self, x, y):
        """
        Returns the matrix of $|x_i-y_j|^2$.
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        cost = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
        return cost

    def sinkhorn_iterations(self, cost):
        """
        Performs Sinkhorn iterations until max_iter
        """
        # Initialise log_alpha
        log_alpha = torch.zeros((cost.shape[0], cost.shape[1]), requires_grad=True).to(cost.device)

        for i in range(self.max_iter):
            log_alpha -= (torch.logsumexp(log_alpha, dim=0, keepdim=True) + torch.logsumexp(log_alpha, dim=1, keepdim=True)) / 2.0
            log_alpha -= torch.log(torch.sum(torch.exp(-cost / self.eps - log_alpha), dim=1, keepdim=True))

        return torch.exp(log_alpha)

    def forward(self, x, y):
        """
        Computes the sinkhorn loss for x and y
        """
        cost = self.cost_matrix(x, y)
        P = self.sinkhorn_iterations(cost)
        loss = torch.sum(P * cost)
        return loss
    
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
        print(x_reshaped.shape)
        # Compute cosine similarity along the second dimension
        similarity = F.cosine_similarity(x_reshaped, y_reshaped, dim=1)

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
        # print(correlation.shape)
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
        self.feature_extractor = torch.nn.Sequential(*list(vgg.features)[:18])
        self.feature_extractor.eval()
        self.mse_loss = torch.nn.MSELoss()

    @property
    def name(self,):
        return "perceptual"

    def forward(self, x, y):
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        return self.mse_loss(x_features, y_features)