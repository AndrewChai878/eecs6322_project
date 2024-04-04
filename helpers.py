import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def psnr(img1:torch.tensor, img2:torch.tensor) -> float:
    '''
    calculates the peak signal to noise ratio between two images
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio    
    @img1: noise free image
    @img2: approximated image
    '''
    mse = torch.nn.functional.mse_loss(img1, img2)
    max_pixel = 1.0
    return (20 * torch.log10(max_pixel / torch.sqrt(mse))).item()

def train(model, optimizer:torch.optim, dims:list[torch.tensor], target:torch.tensor, n_epochs:int, print_step:int=1000):
    '''
    trains the coordx network
        - move the model and tensors to the device beforehand
    @param model: the model to train
    @param optimizer: the optimizer to use
    @param dims: the input tensors containing the coordinates [X, Y, T*]
    @param target: the target tensor    
    @param n_epochs: number of epochs to train for (20k for image, 100k for video)
    @param print_step: the step to print the loss
    '''
    loss_fn = torch.nn.MSELoss()
    for epoch in range(n_epochs+1):
        optimizer.zero_grad()
        out = model(dims)        
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        if epoch % print_step == 0:
            print(f'Iteration: {epoch} | Loss: {loss.item()}')

def accelerated_train(model, optimizer:torch.optim, dims:list[torch.Tensor], target:torch.tensor, n_epochs:int, print_step:int=1000, n_samples:int=262144):
    '''
    trains the coordx network using accelerated sampling
        - move the model and tensors to the device beforehand
    @param model: the model to train
    @param optimizer: the optimizer to use
    @param dims: the input tensors containing the coordinates [X, Y, T*]
    @param target: the target tensor    
    @param n_epochs: number of epochs to train for (20k for image, 100k for video)
    @param print_step: the step to print the loss
    @param n_samples: the (rough) number of pixels to sample
    '''

    v = dims[0].shape[0] * dims[1].shape[0] if len(dims) == 2 else dims[0].shape[0] * dims[1].shape[0] * dims[2].shape[0]
    div = 2 if len(dims) == 2 else 3
    mu = (n_samples/v)**(1/div)
    target = target.to(device)

    x_dist = torch.ones(dims[0].shape[0], device=device)/dims[0].shape[0]
    y_dist = torch.ones(dims[1].shape[0], device=device)/dims[1].shape[0]    
    if len(dims) == 3:
        t_dist = torch.ones(dims[2].shape[0], device=device)/dims[2].shape[0]        
    
    loss_fn = torch.nn.MSELoss()
    for epoch in range(n_epochs+1):
        # Sampling 
        # add a bit of noise 
        mu_noisy = min(random.normalvariate(mu=mu,sigma=0.01),1)

        # use torch multinomial to select coordinates for each dim
        num_samples = int(mu_noisy*dims[0].shape[0])
        x_samples = x_dist.multinomial(num_samples).sort().values

        num_samples = int(mu_noisy*dims[1].shape[0])
        y_samples = y_dist.multinomial(num_samples).sort().values

        if len(dims) == 3:
            num_samples = int(mu_noisy*dims[2].shape[0])
            t_samples = t_dist.multinomial(num_samples).sort().values

        # use torch index_select for grabbing the index of each dim
        if len(dims) == 3:
            sampled_target = target.index_select(0, t_samples).index_select(1, y_samples).index_select(2, x_samples)
        else:
            sampled_target = target.index_select(1, y_samples).index_select(2, x_samples)

        # Training     
        optimizer.zero_grad()
        if len(dims) == 3:
            out = model([dims[0].index_select(0, x_samples), dims[1].index_select(0, y_samples), dims[2].index_select(0, t_samples)])
        else:
            out = model([dims[0].index_select(0, x_samples), dims[1].index_select(0, y_samples)])     
        loss = loss_fn(out, sampled_target)
        loss.backward()
        optimizer.step()
        if epoch % print_step == 0:
            print(f'Iteration: {epoch} | Loss: {loss.item()}')