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
    def __init__(self, n_branches:int, out_channels:int, hidden_size:int, R:int=1, R_strat:int=1, initialize:bool=True):
        '''
        CoordX network as described in the paper: https://arxiv.org/pdf/2201.12425.pdf
        @param branches: number of branches in the network (2 for image, 3 for video)
        @param out_channels: number of output channels (3 for RGB, 1 for grayscale)
        @param hidden_size: hidden size of the network (256 for image, 1024 for video)
        @param R: scale by which to increase hidden feature size (increases image quality; set = 2 for image)
        @param R_strat: strategy for increasing hidden feature size. 
            1 --> increase just the layer before the merge (recommended); 2 --> increase all layers before merge
        @param intialize: whether to initialize the network with Siren initialization
                            - not included in the paper but recommended
        '''
        super(coordx_net, self).__init__()
        assert n_branches in [2,3], "Number of branches must be 2 or 3"
        assert type(R) is int, "R must be an integer"
        self.n_branches = n_branches
        self.channels = out_channels
        self.R = R
        # the scaling factor for the hidden layers before the merge (except the layer immediately before it)
        R_all = 1 if R_strat == 1 else R
        # the input branches
        self.branches = nn.ModuleList()
        for _ in range(n_branches):
            self.branches.append(nn.Linear(1,hidden_size * R_all))
        # the shared hidden layers (premerge)
        self.premerge = nn.Sequential(
            Sine(w0=1.0),
            nn.Linear(hidden_size * R_all, hidden_size * R_all),
            Sine(w0=1.0),
            nn.Linear(hidden_size * R_all, hidden_size * R),
            Sine(w0=1.0)
        )
        # the shared hidden layers (postmerge)
        self.postmerge = torch.nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            Sine(w0=1.0),
            nn.Linear(hidden_size,self.channels),
            nn.Sigmoid()
        )
        # initialize the network with Siren initialization
        if initialize:
            for module in self.branches.modules():
                siren_init(module)            
            for module in self.premerge.modules():
                siren_init(module)
            for module in self.postmerge.modules():
                siren_init(module)
    
    def merge(self, dims:list[torch.tensor]) -> torch.tensor:
        '''
        implementation of the fusion operator in the paper
        @param dims: the input tensors --> [x,y] for image; [x,y,t] for video
        '''
        if self.n_branches == 2:        
            if self.R > 1:
                dims[0] = torch.einsum("ijk->ik", dims[0])
                dims[1] = torch.einsum("ijk->ik", dims[1])
            return torch.einsum("ik,jk->ijk", dims[0], dims[1])
        elif self.n_branches == 3:
            if self.R > 1:
                dims[0] = torch.einsum("ijk->ik", dims[0])
                dims[1] = torch.einsum("ijk->ik", dims[1])
                dims[2] = torch.einsum("ijk->ik", dims[2])
            return torch.einsum("ih,jh,kh->ijkh", dims[0], dims[1], dims[2])
    
    def reshape(self, tensor:torch.tensor) -> torch.tensor:
        '''
        reshape the tensor to be in the correct format for viewing
        @param tensor: the input tensor
        '''
        if self.n_branches == 2:
            # reshape to (C, H, W)
            return tensor.permute(2,1,0)
        elif self.n_branches == 3:
            # reshape to (T, H, W, C)
            return tensor.permute(2,1,0,3)
    
    def map_to_h(self, x:torch.tensor) -> torch.tensor:
        ''' 
        reshapes the input tensor so that its dimensionality is in line with the postmerge hidden layers
        @param x: the input tensor (probably the output of the final premerge layer)
        '''
        if self.R > 1:
            return torch.reshape(x, (x.shape[0], self.R, int(x.shape[1]/self.R)))
        return x

    def forward(self, dims:list[torch.tensor]) -> torch.tensor:
        '''
        Pass the tensors containing each dimensions coordinates as separate inputs
        @param dims: the input tensors containing the coordinates
           - convention: [x,y] for image; [x,y,t] for video
        '''        
        outputs = []
        for i, dim in enumerate(dims):            
            out = self.branches[i](dim)
            out = self.premerge(out)
            out = self.map_to_h(out)
            outputs.append(out)            
        merged = self.merge(outputs)        
        res = self.postmerge(merged)        
        res = self.reshape(res)
        return res