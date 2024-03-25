import torch
import torch.nn as nn
from siren import Sine

def siren_init(module:nn.Module):
    ''' 
    initialization following the Siren paper: https://arxiv.org/pdf/2006.09661.pdf
    initializes W from U(-sqrt(6/fan-in), sqrt(6/fan-in)) where fan-in is the input dimension
    '''
    if isinstance(module, nn.Linear):
        w = module.weight
        # the input dimension of the layer
        fan = w.size(dim=1)
        bound = torch.sqrt(torch.tensor(6.0 / fan))
        with torch.no_grad():
            w.uniform_(-bound, bound)

class coordx_net(nn.Module):
    def __init__(self, channels:int, hidden_size:int=256, initialize:bool=True):
        '''
        CoordX network as described in the paper: https://arxiv.org/pdf/2201.12425.pdf
        @param channels: number of channels in the output image (3 for RGB, 1 for grayscale)
        @param hidden_size: hidden size of the network (default 256)
        @param intialize: whether to initialize the network with Siren initialization
                            - not included in the paper but recommended

        To make a prediction, pass the x and y coordinates as two separate inputs
        '''
        super(coordx_net, self).__init__()
        self.channels=channels
        self.x_first = torch.nn.Linear(1,hidden_size)
        self.y_first = torch.nn.Linear(1,hidden_size)
        self.premerge_parallels = nn.Sequential(
            Sine(w0=1.0),
            nn.Linear(hidden_size,hidden_size),
            Sine(w0=1.0),
            nn.Linear(hidden_size,hidden_size),
            Sine(w0=1.0)
        )
        
        self.postmerge = torch.nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            Sine(w0=1.0),
            nn.Linear(hidden_size,self.channels),
            nn.Sigmoid()
        )
    
        if initialize:
            siren_init(self.x_first)
            siren_init(self.y_first)
            for module in self.premerge_parallels.modules():
                siren_init(module)
            for module in self.postmerge.modules():
                siren_init(module)
    
    def merge(self, x:torch.tensor, y:torch.tensor):
        return torch.einsum("ik,jk->ijk",x,y)

    def forward(self, x:torch.tensor, y:torch.tensor):
        x1 = self.x_first(x)
        y1 = self.y_first(y)
        x2 = self.premerge_parallels(x1)
        y2 = self.premerge_parallels(y1)
        merged = self.merge(x2,y2)
        res = self.postmerge(merged)
        # reshape to (channels, H, W)
        res = res.permute(2,1,0)
        return res