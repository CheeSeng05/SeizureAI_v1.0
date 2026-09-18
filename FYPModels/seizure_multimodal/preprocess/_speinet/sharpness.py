# Adapted from yangt1013/SPEINet; see LICENSE and UPSTREAM.md in this directory.
import torch
import torchvision
import pywt
from ptwt import wavedec2
from torch import Tensor
from torch.nn.functional import lp_pool2d, avg_pool2d, conv2d


def sobel(img: Tensor):
    r'''
    Compute the sobol filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    device = img.device
    Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).to(device)
    Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).to(device)
    G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
    G = G.unsqueeze(1)
    x = conv2d(img, G, padding='same')
    out = torch.sum(x**2, dim=1, keepdim=True).sqrt()
    return out

def laplacian(img: Tensor):
    r'''
    Compute the laplacian filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    # laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    device = img.device
    laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    out = conv2d(img, laplacian_kernel, stride=1, padding=1)
    return out

def mask(img):
    r'''
    Compute the laplacian filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    device = img.device
    mask_size = torch.tensor([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    out = conv2d(img, mask_size)
    return out

def mis3_kernel(img: Tensor):
    r'''
    Compute the sobol filter on a batch of grayscale images
    img : batch_size x 1 x H x W
    '''
    device = img.device
    Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]).to(device)
    Gy = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).to(device)
    grad_x = conv2d(img, Gx.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = conv2d(img, Gy.unsqueeze(0).unsqueeze(0), padding=1)
    out = torch.abs(grad_x) + torch.abs(grad_y)
    return out

def focus_measure_mis3(frames, kernel_size:int):
    device = frames.device
    mis3_filter = torch.zeros((9,1,3,3), device=device)
    mis3_filter[:,:,1,1] = 1
    mis3_filter[0,0,0,0] = -1
    mis3_filter[1,0,0,1] = -1
    mis3_filter[2,0,0,2] = -1
    mis3_filter[3,0,1,0] = -1
    mis3_filter[4,0,1,1] = 0
    mis3_filter[5,0,1,2] = -1
    mis3_filter[6,0,2,0] = -1
    mis3_filter[7,0,2,1] = -1
    mis3_filter[8,0,2,2] = -1
    contrast = conv2d(frames ,mis3_filter, padding=1).abs().sum(dim=1, keepdim=True)
    return lp_pool2d(contrast, norm_type=1, kernel_size=kernel_size).mean(dim=(1,2,3))

def focus_measure_gra7(frames, kernel_size):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    g = sobel(frames)
    g_bar = avg_pool2d(g, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    phi = lp_pool2d((g - g_bar), kernel_size=kernel_size, norm_type=2)**2
    return phi.mean(dim=(1,2,3))

def focus_measure_lap1(frames, kernel_size: int):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    la = laplacian(frames)
    phi = lp_pool2d(la, norm_type=2, kernel_size=kernel_size)**2
    return phi.mean(dim=(1,2,3))

def focus_measure_wave1(frames: Tensor, kernel_size: int):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    wavelet = pywt.Wavelet('db6')
    W_LH1, W_HL1, W_HH1 = wavedec2(frames, wavelet, mode = 'zero', level=1)[1]
    phi = torch.sum((torch.abs(W_LH1)+ torch.abs(W_HL1) + torch.abs(W_HH1)), dim=(1,2,3))
    return phi

def focus_measure_sta3(frames, kernel_size):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    avg = avg_pool2d(frames, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    phi = lp_pool2d((frames - avg), kernel_size=kernel_size, norm_type=2)**2
    return phi.mean(dim=(1,2,3))

def focus_measure_dct3(frames, kernel_size:int):
    r'''
    frames is np array H x W containing a sequence of gray scale images
    '''
    g = mask(frames)
    phi = lp_pool2d(g, kernel_size=kernel_size, norm_type=1)**2
    return phi.mean(dim=(1,2,3))

def generate_vars(frames: Tensor, kernel_size: int, device):
    r'''
    An incredibly long descritpion of this script with the link to the paper
    '''
    frames = frames.to(device)
    gray_frames = torchvision.transforms.Grayscale()(frames)/255.
    lap = focus_measure_lap1(gray_frames, kernel_size)
    mis3 = focus_measure_mis3(gray_frames, kernel_size)
    wave1 = focus_measure_wave1(gray_frames, kernel_size)
    gra_7 = focus_measure_gra7(gray_frames, kernel_size)
    sta_3 = focus_measure_sta3(gray_frames, kernel_size)
    dct_3 = focus_measure_dct3(gray_frames, kernel_size)
    return lap, mis3, wave1, gra_7, sta_3, dct_3
