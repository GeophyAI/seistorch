# from pytorch_msssim import SSIM as _ssim

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from seistorch.signal import differentiable_trvaletime_difference as dtd
from geomloss import SamplesLoss
from seistorch.signal import hilbert, abs, square, integrate, envelope
from ot.lp import wasserstein_1d
import ot

try:
    from torchvision.models import vgg19
except:
    print("Cannot import torchvision.models.vgg19")

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

class Envelope(torch.nn.Module):
    """
    A custom PyTorch module that computes the envelope-based mean squared error loss between
    two input tensors (e.g., predicted and observed seismograms).
    """

    def __init__(self, reduction='sum', criterion='l2'):
        """
        Initialize the parent class.
        """
        super(Envelope, self).__init__()
        self.loss = {'l1':torch.nn.L1Loss(reduction=reduction), 
                     'l2':torch.nn.MSELoss(reduction=reduction)}[criterion]

    @property
    def name(self,):
        return "envelope"

    # @torch.no_grad()
    # def analytic(self, data):
    #     """
    #     Compute the Hilbert transform of the input data tensor.

    #     Args:
    #         data (torch.Tensor): The input data tensor.

    #     Returns:
    #         torch.Tensor: The Hilbert transform of the input data tensor.
    #     """

    #     nt, _, _ = data.shape
    #     # nfft = 2 ** (nt - 1).bit_length()
    #     nfft = nt # the scipy implementation uses this

    #     # Compute the FFT
    #     data_fft = torch.fft.fft(data, n=nfft, dim=0)

    #     # Create the filter
    #     # h = torch.zeros(nfft, device=data.device).unsqueeze(1).unsqueeze(2)
    #     h = np.zeros(nfft, dtype=np.float32)

    #     if nfft % 2 == 0:
    #         h[0] = h[nfft // 2] = 1
    #         h[1:nfft // 2] = 2
    #     else:
    #         h[0] = 1
    #         h[1:(nfft + 1) // 2] = 2

    #     h = np.expand_dims(h, 1)
    #     h = np.expand_dims(h, 2)
    #     h = torch.from_numpy(h).to(data.device)
    #     # h = h.requires_grad_(True)
    #     # Apply the filter and compute the inverse FFT
    #     hilbert_data = torch.fft.ifft(data_fft * h, dim=0)

    #     # Truncate the result to the original length
    #     #hilbert_data = hilbert_data#[:nt]

    #     return hilbert_data
    
    # def envelope(self, seismograms):
    #     """
    #     Compute the envelope of the input seismograms tensor.

    #     Args:
    #         seismograms (torch.Tensor): The input seismograms tensor.

    #     Returns:
    #         torch.Tensor: The envelope of the input seismograms tensor.
    #     """
    #     # Compute the Hilbert transform along the time axis
    #     hilbert_transform = self.analytic(seismograms)
        
    #     # Compute the envelope
    #     envelope = torch.abs(hilbert_transform)

    #     # envelope = torch.sqrt(hilbert_transform.real**2 + hilbert_transform.imag**2)

    #     return envelope

    # def envelope_loss(self, pred_seismograms, obs_seismograms):
    #     """
    #     Compute the envelope-based mean squared error loss between pred_seismograms and obs_seismograms.

    #     Args:
    #         pred_seismograms (torch.Tensor): The predicted seismograms tensor.
    #         obs_seismograms (torch.Tensor): The observed seismograms tensor.

    #     Returns:
    #         torch.Tensor: The computed envelope-based mean squared error loss.
    #     """
        
    #     pred_envelope = self.envelope(pred_seismograms)**2
    #     obs_envelope = self.envelope(obs_seismograms)**2

    #     # pred_envelope = pred_envelope/(torch.norm(pred_envelope, p=2, dim=0)+1e-16)
    #     # obs_envelope = obs_envelope/(torch.norm(obs_envelope, p=2, dim=0)+1e-16)

    #     # loss = F.mse_loss(pred_envelope, obs_envelope, reduction="mean")
    #     return torch.nn.MSELoss(reduction='sum')(pred_envelope, obs_envelope)

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
            #loss += self.envelope_loss(x[i], y[i])
            loss += self.loss(envelope(x[i])**2, envelope(y[i])**2)
        return loss

class ImplicitLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features.cuda()
        # self.feature_extractor = torch.nn.Sequential(*list(vgg.features)[:35]).cuda()
        #self.feature_extractor.eval()

        # Define the downsampling layer
        self.pool = torch.nn.AvgPool2d(kernel_size=(4, 4), stride=(2, 2))

    @property
    def name(self,):
        return "implicit"

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0':'conv1_1',
                      '5':'conv2_1',
                      '10':'conv3_1',
                      '19':'conv4_1',
                      '21':'conv4_2', # content repr
                      '28':'conv5_1',}

        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
            
        return features

    def to_rgb(self, x):
        x_mono = x.mean(dim=-1, keepdim=True)  # Take the mean along the channel dimension to get a single-channel image
        
        x_rgb = x_mono.repeat(1, 1, 1, 3)  # Repeat the single-channel image along the channel dimension to get a three-channel image
        
        x_rgb = x_rgb.permute(0, 3, 1, 2)
        # Apply the pooling operation to downsample the data
        x_rgb = self.pool(x_rgb)

        return x_rgb

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h*w)
        return torch.mm(tensor, tensor.t()) #gram
    
    def forward(self, x, y):

        # Normalize along the batch dimension
        # x = x / torch.max(torch.abs(x), dim=0, keepdim=True).values
        # y = y / torch.max(torch.abs(y), dim=0, keepdim=True).values

        # Compute the feature maps
        x = self.to_rgb(x)
        y = self.to_rgb(y)

        # get features
        x_features = self.get_features(x, self.vgg)
        y_features = self.get_features(y, self.vgg)

        # calculate content loss
        content_loss = torch.mean((x_features['conv4_2'] - y_features['conv4_2'])**2)
        # calculate grams
        style_grams = {layer:self.gram_matrix(y_features[layer]) for layer in y_features}

        # weights for style layers 
        style_weights = {'conv1_1':1.,
                        'conv2_1':1.,
                        'conv3_1':1.,
                        'conv4_1':1.,
                        'conv5_1':1.}
		# calculate style loss
        style_loss = 0

        for layer in style_weights:
            target_feature = x_features[layer]
            _, d, h, w = target_feature.shape

            # target gram
            target_gram = self.gram_matrix(target_feature)

            # style gram
            style_gram = style_grams[layer]

            # style loss for curr layer
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

            style_loss += layer_style_loss / (d * h * w)

        return content_loss#+style_loss

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
        return torch.nn.L1Loss(reduction='sum')(x, y)
    
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

class NormalizedIntegrationMethod(torch.nn.Module):
    """Donno et.al, doi: 10.3997/2214-4609.20130411
    """
    def __init__(self, criterion='l2', reduction='sum'):
        super(NormalizedIntegrationMethod, self).__init__()
        self.loss = {'l1':torch.nn.L1Loss(reduction=reduction), 
                     'l2':torch.nn.MSELoss(reduction=reduction)}[criterion]

    @property
    def name(self,):
        return "nim"
    
    def forward(self, x, y):
        # ensure non-negative eq. (1.2)
        x = square(x) 
        y = square(y)

        # weights of each trace
        wx = torch.sum(x, dim=1, keepdim=True)[0]
        wy = torch.sum(y, dim=1, keepdim=True)[0]

        # intergration # eq. (1.2)
        x = integrate(x) 
        y = integrate(y)

        # ensure equals to one at the end
        # the denominator in equation (1.2) is wrong
        # x = x/wx
        # y = y/wy
        # the denominator should be the maximum value of the trace
        # so that the value at the end is one
        x = x / x.max() 
        y = y / y.max()


        # compute the loss
        loss = self.loss(x, y)
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

class Wasserstein1d(torch.nn.Module):
    # TODO
    # Implement this by hand(refer to 
    # /root/miniconda3/lib/python3.8/site-packages/ot/lp/solver_1d.py)
    def __init__(self):
        super(Wasserstein1d, self).__init__()

    @property
    def name(self,):
        return "w1d"
    
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
        x = x**2
        y = y**2
        # Enforce sum to one on the support
        # Method 1: normalize use the sum value
        # wx = torch.sum(x, dim=0, keepdim=True)[0]
        # wy = torch.sum(y, dim=0, keepdim=True)[0]
        # x = x / wx
        # y = y / wy
        # Compute the Wasserstein distance
        loss = 0
        t = torch.linspace(0, nt-1, nt, dtype=x[0].dtype).to(x.device)*0.001#[None,:]
        # t /= t.max()
        # loss matrix and normalization

        for _x, _y in zip(x, y):

            # Ensure nt last
            # _x = _x.permute(1, 2, 0).view(-1, nt).contiguous()
            # _y = _y.permute(1, 2, 0).view(-1, nt).contiguous()
            # loss += wasserstein_1d(t, t, _x, _y, p=2).mean()
            for r in range(nr):
                for c in range(nc):
                    # px = torch.cumsum(_x[:, r, c], dim=0)
                    # py = torch.cumsum(_y[:, r, c], dim=0)
                    px = _x[:, r, c]
                    py = _y[:, r, c]

                    # normalize


                    if torch.sum(px)==0 or torch.sum(py)==0:
                        loss +=0.
                    else:

                        # px = px / torch.sum(px)
                        # py = py / torch.sum(py)
                        #loss += ot.wasserstein_1d(px, py, p=2, require_sort=False)
                        loss += ot.wasserstein_1d(t, t, px, py, p=2, require_sort=True)
                        #loss += ot.emd2_1d(t, t, px, py, p=2)
                        # euclidean

                        # M = ot.dist(py.view((nt, 1)), py.view((nt, 1)), 'euclidean')
                        # M /= M.max() * 0.1

                        # d_emd = ot.emd2(px, py.view(nt, 1), M)

                        # sqeuclidean
                        # M2 = ot.dist(py.reshape((nt, 1)), py.reshape((nt, 1)), 'sqeuclidean')
                        # M2 /= M2.max() * 0.1

                        # d_emd = ot.emd2(px, py.view(nt, 1), M2)

                        # loss += d_emd[0]

            #             # loss += wasserstein_1d(t, t, px, py, p=1)

        return loss
