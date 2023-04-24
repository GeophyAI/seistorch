import torch

class NormalizedCrossCorrelation(torch.nn.Module):
    def __init__(self, dt=1.0):
        self.dt = dt
        super().__init__()
    
    def forward(self, x, y):
        loss = 0
        for _x, _y in zip(x, y):
            normx = torch.linalg.norm(_x, ord=2, axis=0)
            normy = torch.linalg.norm(_y, ord=2, axis=0)
            nx = _x/(normx+1e-12)
            ny = _y/(normy+1e-12)
            loss+=torch.mean(torch.pow(nx-ny, 2))

            #for i in range(_x.size(1)):
            #    loss+=torch.mean(torch.pow(torch.dot(nx[:,i,0], ny[:,i,0])*nx[:,i,0]-ny[:,i,0], 2))
        return loss

    
class ElasticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        loss = 0
        for _x, _y in zip(x, y):
            loss += torch.mean(torch.pow(_x-_y, 2))
        return loss