import torch

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