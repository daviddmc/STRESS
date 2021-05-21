from scipy.special import i0, i1
from scipy.stats import trim_mean
import torch
import numpy as np
import os
from skimage.metrics import structural_similarity

class MovingAverage:
    def __init__(self, alpha):
        assert 0 <= alpha < 1
        self.alpha = alpha
        self.value = dict()

    def __call__(self, key, value):
        if key not in self.value:
            self.value[key] = (0, 0)
        num, v = self.value[key]
        num += 1
        if self.alpha:
            v = v * self.alpha + value * (1 - self.alpha)
        else:
            v += value
        self.value[key] = (num, v)
    
    def __str__(self):
        s = ''
        for key in self.value:
            num, v = self.value[key]
            if self.alpha:
                s += "%s = %f\t" % (key, v / (1 - self.alpha**num))
            else:
                s += "%s = %f\t" % (key, v / num)
        return s

def rician_correct(out, sigma, background):
    if sigma == 0:
        out[out < 0] = 0
        return out
    elif sigma is None:
        sigma_pi = trim_mean(out[background].cpu().numpy(), 0.1, None)
        sigma = sigma_pi * np.sqrt(2/np.pi)
    else:
        sigma_pi = sigma * np.sqrt(np.pi/2)
    
    old_out = out
    out = out / sigma_pi
    out[out < 1] = 1
    curVal=0
    for coeff in [-0.02459419,  0.28790799,  0.27697441,  2.68069732]:
        curVal = (curVal+coeff)*out
    out = (curVal - 3.22092921) * (sigma**2)
    snr_mask = old_out/sigma > 3.5
    out[snr_mask] = old_out[snr_mask]**2 - sigma**2
    out = torch.sqrt(out)
    return out

def fba(imgs, p):
    freqs = [np.fft.rfftn(img) for img in imgs]
    weights = [np.abs(freq) ** p for freq in freqs]
    return np.fft.irfftn(sum(freq * weight for freq, weight in zip(freqs, weights)) / sum(weights)).astype(np.float32)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def psnr(x, y, mask=None):
    if mask is None:
        mse = np.mean((x - y) ** 2)
    else:
        mse = np.sum(((x - y) ** 2) * mask) / mask.sum() 
    return 10 * np.log10(y.max()**2 / mse)

def ssim(x, y, mask=None):
    mssim, S = structural_similarity(x, y, full=True)
    if mask is not None:
        return (S * mask).sum() / mask.sum()
    else:
        return mssim

def ssim_slice(x, y, mask):
    mask = mask.sum((0,1)) > 0
    #print(np.nonzero(mask))
    x = x[..., mask]
    y = y[..., mask]

    return structural_similarity(x, y)
    #ssims = []
    #for i in range(x.shape[-1]):
    #    ssims.append(structural_similarity(x[..., i], y[..., i]))
    #return np.mean(ssims)

    