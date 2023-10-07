# from pytorch_msssim import SSIM as _ssim

import numpy as np
import torch
import torch.nn.functional as F
from seistorch.signal import travel_time_diff
from geomloss import SamplesLoss

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
        _, nt, nr, nc = x.shape
        for i in range(x.shape[0]):
            _x = x[i]
            _y = y[i]
            # Reshape x and y to (time_samples, num_traces * num_channels)
            x_reshaped = _x.view(nt, -1)
            y_reshaped = _y.view(nt, -1)
            # Compute cosine similarity along the ? dimension
            similarity = F.cosine_similarity(x_reshaped, y_reshaped, dim=0, eps=1e-10)
            # Compute the mean difference between similarity and 1
            loss += torch.mean(1-similarity)

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
        loss = 0
        for i in range(x.shape[0]):
            loss += self.envelope_loss(x[i], y[i])
        # loss = self.envelope_loss(x, y)
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
        Compute the envelope-based mean squared error loss for the given input tensors x and y.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The computed envelope-based mean squared error loss.
        """
        loss = 0
        for i in range(x.shape[0]):
            loss += self.phase_consistency(self.analytic(x[i]), self.analytic(y[i]))
        return loss

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
        x = x**2
        y = y**2

        wx = torch.sum(x, dim=1, keepdim=True)[0]
        wy = torch.sum(y, dim=1, keepdim=True)[0]

        x = x / wx
        y = y / wy

        loss = 0
        for i in range(x.shape[0]):
            loss += SamplesLoss("sinkhorn", p=2, blur=0.00005)(x[i], y[i]).mean()
        return loss

class TravelTimeCrossCorrelation(torch.nn.Module):
    def __init__(self, dt=0.001, scale=100):
        """
        Initialize the parent class.
        """
        self.dt = dt
        self.scale = scale
        super(TravelTimeCrossCorrelation, self).__init__()

    @property
    def name(self,):
        return "ttcc"
    
    def forward(self, x, y):
        nb, nt, nr, nc = x.shape
        padding = nt - 1
        loss = 0.
        indices = torch.arange(2*nt-1, device=x.device)
        for b in range(nb):
            for r in range(nr):
                for c in range(nc):
                    _x = x[b,:,r,c]
                    _y = y[b,:,r,c]

                    if torch.max(torch.abs(_x)) >0 and torch.max(torch.abs(_y)) >0:

                        cc = F.conv1d(_x.unsqueeze(0), _y.unsqueeze(0).unsqueeze(0), padding=padding)
                        logits = F.gumbel_softmax(cc*self.scale, tau=1, hard=True)
                        max_index = torch.sum(indices * logits)
                        loss += (max_index-nt+1)*self.dt

                    else:
                        loss += 0.
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
        nb, nt, nr, nc = x.shape
        for b in range(nb):
            for r in range(nr):
                for c in range(nc):
                    loss += travel_time_diff(x[b,:,r,c], y[b,:,r,c], dt)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        syn, obs = ctx.saved_tensors
        adj = torch.zeros_like(syn)
        dt = 0.001#ctx.cfg['geom']['dt']

        adj[:,1:-1,...] = (syn[:,2:,...]-syn[:,:-2,...])/(2.*dt)
        nb, nt, nr, nc = syn.shape
        for b in range(nb):
            for r in range(nr):
                for c in range(nc):
                    tt_diff = travel_time_diff(syn[b,:,r,c], obs[b,:,r,c], dt)
                    norm = torch.sum(adj[b,:,r,c] * adj[b,:,r,c]) * dt
                    adj[b,:,r,c] /= (norm+1e-16)
                    adj[b,:,r,c] *= tt_diff

        adj = adj*grad_output

        return adj, None