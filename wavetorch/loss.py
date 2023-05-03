import torch
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
# from scipy.signal import hilbert

loss_dict =  ["wd", "ncc", "mse", "envelope", "cc", "huber", "phase", "l2", "l1"]
class Loss:

    def __init__(self, loss="mse"):
        assert loss in loss_dict, f"Cannot find loss named {loss}"
        self.loss_name = loss

    def loss(self,):
        return self.__getattribute__(self.loss_name)()
    
    def l2(self,):
        return torch.nn.MSELoss()
    
    def l1(self,):
        return torch.nn.L1Loss()

    def wd(self,):
        return Wasserstein()

    def ncc(self,):
        return NormalizedCrossCorrelation()
    
    def mse(self,):
        return torch.nn.MSELoss()
    
    def envelope(self,):
        return Envelope()
    
    def cc(self,):
        return CrossCorrelation()
    
    def phase(self,):
        return Phase()
    
    def huber(self,):
        return Huber()


class NormalizedCrossCorrelation(torch.nn.Module):
    def __init__(self, dt=1.0):
        """
        Initializes the NormalizedCrossCorrelation module.

        Args:
            dt (float): The time step. Default is 1.0.
        """
        self.dt = dt
        super().__init__()

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

class ElasticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        loss = 0
        for _x, _y in zip(x, y):
            loss += torch.mean(torch.pow(_x-_y, 2))
        return loss

class Wasserstein(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Calculates the Wasserstein distance between two distributions.

        Args:
            x (torch.Tensor): The first distribution.
            y (torch.Tensor): The second distribution.

        Returns:
            torch.Tensor: The Wasserstein distance between the distributions.
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

    def hilbert(self, data):
        nt, ntraces, _ = data.shape
        data = data.squeeze(dim=2)
        nfft = 2 ** (nt - 1).bit_length()

        data_fft = torch.fft.fft(data, n=nfft, dim=0)

        h = torch.zeros(nfft, device=data.device).unsqueeze(1)
        h[0] = 1
        h[1:(nfft // 2)] = 2
        if nfft % 2 == 0:
            h[nfft // 2] = 1

        hilbert_data_fft = data_fft * h
        hilbert_data = torch.fft.ifft(hilbert_data_fft, dim=0)

        hilbert_data = hilbert_data[:nt].view(nt, ntraces, 1)

        return hilbert_data.real

    def instantaneous_phase(self, seismograms):
        hilbert_transform = self.hilbert(seismograms)

        analytic_signal = seismograms + hilbert_transform.to(seismograms.device) * 1j
        phase = torch.angle(analytic_signal)
        return phase

    def forward(self, x, y):
        pred_phase = self.instantaneous_phase(x)
        obs_phase = self.instantaneous_phase(y)

        loss = F.mse_loss(pred_phase, obs_phase)
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

    def hilbert(self, data):
        """
        Compute the Hilbert transform of the input data tensor.
        
        Args:
            data (torch.Tensor): The input data tensor.
        
        Returns:
            torch.Tensor: The Hilbert transform of the input data tensor.
        """
        nt, ntraces, _ = data.shape
        data = data.squeeze(dim=2)
        nfft = 2 ** (nt - 1).bit_length()

        # Compute the FFT
        data_fft = torch.fft.fft(data, n=nfft, dim=0)

        # Create the filter
        h = torch.zeros(nfft, device=data.device).unsqueeze(1)
        h[0] = 1
        h[1:(nfft // 2)] = 2
        if nfft % 2 == 0:
            h[nfft // 2] = 1

        # Apply the filter and compute the inverse FFT
        hilbert_data_fft = data_fft * h
        hilbert_data = torch.fft.ifft(hilbert_data_fft, dim=0)

        # Truncate the result to the original length and reshape
        hilbert_data = hilbert_data[:nt].view(nt, ntraces, 1)

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
        analytic_signal = seismograms + hilbert_transform.to(seismograms.device) * 1j
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
        pred_normalized = pred_seismograms / (pred_seismograms.norm(dim=0) + 1e-8)
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

    def forward(self, x, y):
        loss = F.smooth_l1_loss(x, y, reduction='mean', beta=self.delta)
        return loss

class WaveformDifference(torch.nn.Module):
    def __init__(self):
        super(WaveformDifference, self).__init__()

    def forward(self, observed, modeled):
        # Reshape tensors to (batch_size, 1, sequence_length) for DTW calculation
        observed = observed.unsqueeze(1)
        modeled = modeled.unsqueeze(1)

        # Calculate pairwise distances between observed and modeled waveforms
        distances = pairwise_distance(observed, modeled)

        # Apply the dynamic time warping (DTW) algorithm to find the minimum distance
        dtw_distance = self.dtw(distances)

        # Calculate the mean DTW distance as the loss
        loss = torch.mean(dtw_distance)
        return loss

    def dtw(self, distances):
        batch_size, _, sequence_length = distances.size()

        # Initialize the DTW matrix
        dtw_matrix = torch.zeros((batch_size, sequence_length, sequence_length), device=distances.device)

        # Fill the first row and first column of the DTW matrix
        dtw_matrix[:, 0, 0] = distances[:, 0, 0]
        for i in range(1, sequence_length):
            dtw_matrix[:, i, 0] = distances[:, i, 0] + dtw_matrix[:, i-1, 0]
            dtw_matrix[:, 0, i] = distances[:, 0, i] + dtw_matrix[:, 0, i-1]

        # Fill the rest of the DTW matrix
        for i in range(1, sequence_length):
            for j in range(1, sequence_length):
                dtw_matrix[:, i, j] = distances[:, i, j] + torch.min(
                    dtw_matrix[:, i-1, j],
                    dtw_matrix[:, i, j-1],
                    dtw_matrix[:, i-1, j-1]
                )

        # Return the minimum distance along the bottom-right diagonal of the DTW matrix
        dtw_distance = dtw_matrix[:, sequence_length-1, sequence_length-1]
        return dtw_distance

