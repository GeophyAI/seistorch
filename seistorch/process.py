
from seistorch.signal import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import torch


class PostProcess:

    'A class for post-processing the results of a model.'

    def __init__(self, model, cfg, commands=[]):
        self.model = model

        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module

        self.cfg = cfg
        self.commands = commands

        self.ndim = self.model.cell.geom.ndim

        if self.commands.grad_cut:
            self.modelmask = self.load_seabed()

    def load_seabed(self, ):
        # Parameters
        padding = self.cfg['geom']['boundary']['width']
        multiple = self.cfg['geom']['multiple']
        # Load seabed from disk
        seabed = np.load(self.cfg['geom']['seabed'])
        seabed = torch.from_numpy(seabed).float()
        # Pad the seabed
        top = 0 if multiple else padding
        
        if self.ndim==2: pads = (padding, padding, top, padding)
        if self.ndim==3: pads = (padding, padding, top, padding, padding, padding)
        
        mask = torch.nn.functional.pad(seabed, pads, mode='constant', value=0)
        return mask

    def cut_gradient(self, ):

        for para in self.model.parameters():
            if para.requires_grad:
                para.grad = para.grad * self.modelmask.to(para.device)

    def repad(self,):
            
        padding = self.cfg['geom']['boundary']['width']
        multiple = self.cfg['geom']['multiple']
        top = 0 if multiple else padding

        if self.ndim == 2:
            # [[top, self.bwidth], [self.bwidth, self.bwidth]]
            pads = [[top, padding], [padding, padding]]
        if self.ndim == 3:
            raise NotImplementedError("3D padding not implemented")

        for para in self.model.parameters():
            tensor = para.detach().cpu().numpy()[top:-padding, padding:-padding]
            padded = np.pad(tensor, pads, mode='edge')
            para.data = torch.from_numpy(padded).to(para.device)

    def smooth_gradient(self, ):

        smooth_cfg = self.cfg['training']['smooth']
        counts = smooth_cfg['counts']
        sigma = smooth_cfg['sigma']
        radius = smooth_cfg['radius']

        axis2d = {'z': 0, 'x': 1}
        axis3d = {'x': 0, 'z': 1, 'y': 2}

        for para in self.model.parameters():
            if para.requires_grad:
                grad = para.grad.cpu().detach()#.numpy()
                grad = grad * self.modelmask#.to(para.device)
                for _ in range(counts):
                    if para.ndim == 2:
                        # Smooth along the z axis
                         grad = gaussian_filter(grad, 
                                                     sigma['z'], 
                                                     radius['z'], 
                                                     axis=axis2d['z'])
                         # Smooth along the x axis
                         grad = gaussian_filter(grad, 
                                                     sigma['x'], 
                                                     radius['x'], 
                                                     axis=axis2d['x'])
                        #gaussian_filter1d(grad, sigma['z'], axis=axis2d['z'], radius=radius['z'], output=grad)
                        #gaussian_filter1d(grad, sigma['x'], axis=axis2d['x'], radius=radius['x'], output=grad)
                    elif para.ndim == 3:
                        # Smooth along the x axis
                        raise NotImplementedError("3D smoothing not implemented")
                        # para.grad = gaussian_filter(para.grad, 
                        #                             sigma['x'], 
                        #                             radius['x'], 
                        #                             axis=axis3d['x'])
                        # # Smooth along the z axis
                        # para.grad = gaussian_filter(para.grad, 
                        #                             sigma['z'], 
                        #                             radius['z'], 
                        #                             axis=axis3d['z'])
                        # # Smooth along the y axis
                        # para.grad = gaussian_filter(para.grad, 
                        #                             sigma['y'], 
                        #                             radius['y'], 
                        #                             axis=axis3d['y'])
                para.grad.data = grad.to(para.device)#torch.from_numpy(grad).to(para.device)
