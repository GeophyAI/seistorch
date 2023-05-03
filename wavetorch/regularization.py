import torch
import torch.nn as nn

class TikhonovRegularization(nn.Module):
    def __init__(self, alpha=1.0):
        super(TikhonovRegularization, self).__init__()
        self.alpha = alpha

    def forward(self, model):
        gradient = torch.gradient(model)
        regularization_term = self.alpha * (torch.sum(torch.tensor(gradient) ** 2))
        return regularization_term

class TVRegularization(nn.Module):
    def __init__(self, alpha=1.0):
        super(TVRegularization, self).__init__()
        self.alpha = alpha

    def forward(self, model):
        gradient = torch.gradient(model)
        regularization_term = self.alpha * torch.sum(torch.sqrt(torch.tensor(gradient) ** 2 + 1e-8))
        return regularization_term
    
class LaplacianRegularization(nn.Module):
    def __init__(self, alpha=1.0):
        super(LaplacianRegularization, self).__init__()
        self.alpha = alpha

    def forward(self, model):
        laplacian = torch.gradient(torch.gradient(model))
        regularization_term = self.alpha * torch.sum(torch.tensor(laplacian) ** 2)
        return regularization_term