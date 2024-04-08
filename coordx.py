import math
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

def positional_encoding(tensor:torch.tensor, max_n:int) -> torch.tensor:
    '''
    positional encoding from CoordX paper. It uses the encoding as defined by NERF but additonaly concatenates the original input
    formula: PE(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p), sin(2^1 * pi * p), cos(2^1 * pi * p), ..., sin(2^L * pi * p), cos(2^L * pi * p)]
        - L is the encoding frequency, it is not specified in the CoordX paper but mentioned to be 10 in the NERF paper
    @param tensor: the input tensor
    @param max_n: the maximum value that can appear in the input tensor (needs to be supplied since we are sampling from a distribution and not a fixed range)
    @output: tensor of shape (N, 2*L + 1), where N is the number of samples, L is the encoding frequency (10)
    '''    
    # normalize to [-1, 1]
    nvals = torch.clone(tensor)    
    nvals = 2 * ((nvals - 1) / (max_n - 1)) - 1 # 1 is the min
    # apply the positional encoding to the input tensor
    out = nvals.repeat(1, 20)
    for i in range(10):
        out[:,2*i] = torch.sin(math.pow(2,i) * math.pi * out[:,2*i])
        out[:,2*i+1] = torch.cos(math.pow(2,i) * math.pi * out[:,2*i+1])
    # concatenate this encoding with the original input
    out = torch.cat((out, tensor), dim=1)
    return out    

class coordx_net(nn.Module):
    def __init__(self, n_branches:int, out_channels:int, hidden_size:int, R:int=1, R_strat:int=1, initialize:bool=True, PE:bool=False, maxes:list[int]=[]):
        '''
        CoordX network as described in the paper: https://arxiv.org/pdf/2201.12425.pdf
        @param branches: number of branches in the network (2 for image, 3 for video)
        @param out_channels: number of output channels (3 for RGB, 1 for grayscale)
        @param hidden_size: hidden size of the network (256 for image, 1024 for video)
        @param R: scale by which to increase hidden feature size (increases image quality; set = 2 for image)
        @param R_strat: strategy for increasing hidden feature size. 
            - 1 --> increase just the layer before the merge (recommended)
            - 2 --> increase all layers before merge
        @param intialize: whether to initialize the network with Siren initialization
            - not included in the paper but recommended for photos
        @param PE: whether to include positional encoding (recommended for video)
            - if True then initialize should = False
        @param maxes: **only for PE=True** the maximum values that can appear in the input tensor, shape = [x_max, y_max, t_max*]
        '''
        super(coordx_net, self).__init__()
        assert n_branches in [2,3], "Number of branches must be 2 or 3"
        assert type(R) is int, "R must be an integer"
        self.n_branches = n_branches
        self.channels = out_channels
        self.R = R
        self.PE = PE
        self.maxes = maxes
        # the scaling factor for the hidden layers before the merge (except the layer immediately before it)
        R_all = 1 if R_strat == 1 else R
        # the input branches
        self.branches = nn.ModuleList()
        # the activation function
        self.activation = nn.ReLU() if self.PE else Sine(w0=1.0)
        for _ in range(n_branches):
            if self.PE:
                self.branches.append(nn.Linear(2*10 + 1, hidden_size * R_all))
            else:
                self.branches.append(nn.Linear(1, hidden_size * R_all))
        # the shared hidden layers (premerge)
        self.premerge = nn.Sequential(
            self.activation,            
            nn.Linear(hidden_size * R_all, hidden_size * R_all),            
            self.activation,
            nn.Linear(hidden_size * R_all, hidden_size * R),
            self.activation            
        )
        # the shared hidden layers (postmerge)
        self.postmerge = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            self.activation,
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
        if self.PE:
            for i in range(len(dims)):
                dims[i] = positional_encoding(dims[i], self.maxes[i])
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