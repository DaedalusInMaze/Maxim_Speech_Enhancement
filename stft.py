import torch

import torch.nn as nn

import torch.nn.functional as F

import librosa

from utils import quantize_spectrum, normalize_quantized_spectrum


def irm(clean_mag, noise_mag):
    """
    ideal ratio mask

    to recover: predicted mask * noisy mag = clean mag
    """
    eps= 1e-8
    return (clean_mag ** 2 / (clean_mag ** 2 + noise_mag ** 2 + eps)) ** 0.5


def psm(clean_fft, mixed_fft):
    """
    phase sensitive mask
    """
    return torch.abs(clean_fft) / torch.abs(mixed_fft) * torch.cos(torch.angle(mixed_fft) + torch.angle(clean_fft))


def logmag_transform(magnitude, recover = False):
    
    if recover:
        
        magnitude = torch.expm1(magnitude)
            
        magnitude = torch.clamp(magnitude, min= 0, max= torch.amax(magnitude))
        
        return magnitude
    
    return torch.log1p(magnitude)


def lps_transform(magnitude, recover = False):
    """
    log power spectrum
    """
    
    if recover:
        
        magnitude = torch.sqrt(10 ** magnitude)
        
        return magnitude
    
    return torch.log10(magnitude ** 2)



def quantize_transform(magnitude, recover = False):
    
    if recover:
        
        magnitude = torch.clamp(magnitude, min= 0, max= torch.amax(magnitude))
        
        magnitude = normalize_quantized_spectrum(magnitude)
        
        return magnitude
    
    magnitude = magnitude / torch.amax(magnitude)
    
    return quantize_spectrum(magnitude)

class STFT(nn.Module):
    
    def __init__(self, n_fft, hop_len, win_len, window, transform_type='logmag'):
        """
        - n_fft: 计算FFT的点数， 越大越细致，但要求更大计算能力，最好是power of 2，
        - hop_length：窗的移动距离
        - win_length：窗的大小
        - window：hanning
        - device: cuda or cpu
        - train: 训练还是预测
        """
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.window = window
        self.transform_type = transform_type
    
    def forward(self, dt):

        with torch.no_grad():
            device = dt['mixed'].device

            for key in ['mixed', 'clean', 'noise']:

                x_stft = torch.tensor(librosa.stft(
                    dt[key].cpu().numpy().squeeze(), 
                    n_fft= self.n_fft, 
                    win_length= self.win_len, 
                    hop_length= self.hop_len, 
                    window= self.window)).T

                mag = torch.abs(x_stft).to(device)
                if key == 'mixed':
                    dt['phase'] = torch.exp(1j * torch.angle(x_stft)).to(device)

                
                if self.transform_type == 'logmag':
                    dt[f'{key}_mag'] = logmag_transform(mag)
                elif self.transform_type == 'lps':
                    dt[f'{key}_mag'] = lps_transform(mag)
                else:
                    dt[f'{key}_mag'] = quantize_transform(mag)

            dt['mask'] = torch.div(dt['clean_mag'], dt['mixed_mag'])            

        return dt


class ISTFT(nn.Module):

    def __init__(self, hop_len, win_len, window, device, chunk_size, transform_type='logmag'):
        """
        和STFT 参数一样
        """
        super(ISTFT, self).__init__()
        self.hop_len = hop_len
        self.win_len = win_len
        self.window = window
        self.chunk_size = chunk_size
        self.device = device
        self.transform_type = transform_type
        self.flag = False

    def forward(self, dt):
        
        # dt['pred_y'] = self.subtraction(pred= dt['pred_y'], noisy_mag= dt['mixed_mag'])#predict noise
        
        # dt['pred_y'] = torch.clamp(dt['pred_y'], min= 0, max= torch.amax(dt['pred_y']))
                                        
        dt['pred_y'] = self.recovery(mag= dt['pred_y'], phase= dt['phase'])
        
        dt['true_y'] = self.recovery(mag= dt['clean_mag'][self.chunk_size: ], phase= dt['phase'])
        
        if not self.flag:
            
            dt['mixed_y'] = self.recovery(mag= dt['mixed_mag'][self.chunk_size: ], phase= dt['phase'])
            
            dt['mixed_y'] = librosa.istft(
                dt['mixed_y'].cpu().detach().numpy().T,
                hop_length=self.hop_len,
                win_length=self.win_len,
                window=self.window,
                center=True,
                dtype=None,
                length=None)
            
            self.flag = True
        
        for key in ['pred_y', 'true_y']:
            dt[key] = librosa.istft(
                            dt[key].cpu().detach().numpy().T,
                            hop_length=self.hop_len,
                            win_length=self.win_len,
                            window=self.window,
                            center=True,
                            dtype=None,
                            length=None)
        return dt
    
    def recovery(self, mag, phase):
        
        if self.transform_type == 'logmag':
            mag = logmag_transform(mag, recover=True)
            
        elif self.transform_type == 'lps':
            mag = lps_transform(mag, recover=True)
        else:
            mag = quantize_transform(mag, recover=True)
    
        a = mag.cpu()
        b = torch.cos(phase[self.chunk_size : ]) + 1j * torch.sin(phase[self.chunk_size : ]).cpu()
        return a * b
    
    def subtraction(self, pred, noisy_mag):
        res = noisy_mag[self.chunk_size: ] - pred
        res = torch.clamp(res, min= 0, max= torch.amax(res))
        return res
        



        
class torch_stft(nn.Module):
    
    def __init__(self, n_fft, hop_length, win_length, device, transform_type='logmag'):
        
        super(torch_stft, self).__init__()
        
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length= win_length
        self.window = torch.hann_window(n_fft, device= device)
        self.device = device
        self.transform_type = transform_type
    
    def forward(self, dt):
        
        for key in ['mixed', 'clean', 'noise']:

            fft = torch.stft(dt[key],
                             n_fft = self.n_fft,
                             hop_length=self.hop_length,
                             win_length=self.win_length,
                             window=self.window,
                             return_complex=True).T
            fft = torch.squeeze(fft)
            mag = torch.abs(fft)
            
            if key == 'mixed':
                dt['phase'] = torch.exp(1j * torch.angle(fft))

            if self.transform_type == 'logmag':
                dt[f'{key}_mag'] = logmag_transform(mag)
            elif self.transform_type == 'lps':
                dt[f'{key}_mag'] = lps_transform(mag)
            else:
                dt[f'{key}_mag'] = quantize_transform(mag)

        dt['mask'] = irm(dt['clean_mag'], dt['noise_mag'])
            
        return dt
    

class torch_istft(nn.Module):
    
    def __init__(self, n_fft, hop_length, win_length, chunk_size, device, target, transform_type='logmag', cnn='1d'):
        
        super(torch_istft, self).__init__()
        
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length= win_length
        self.window = torch.hann_window(n_fft, device= 'cpu')
        self.device = device
        self.transform_type = transform_type
        self.chunk_size = chunk_size
        self.cnn = cnn
        self.target = target
    
    def mask_recover(self, dt):
        
        batch, time, freq = dt['pred_y'].shape
        
        if freq == 256:
            
            dt['pred_y'] = F.pad(dt['pred_y'], (0, 1, 0, 0))
            
            freq += 1
        
        dt['pred_y'] = torch.reshape(dt['pred_y'], (-1, freq))
        
        lens = dt['pred_y'].shape[0]
        
        dt['pred_y'] = dt['pred_y'] * dt['mixed_mag'][:lens]
        
        dt['pred_y'] = torch.reshape(dt['pred_y'], (batch, time, freq))
        
        return dt
    
    def cnn1d_recover(self, dt):
        
        dt['pred_y'] = torch.multiply(dt['pred_y'], dt['phase'][self.chunk_size : ])

        dt['true_y'] = torch.multiply(dt['clean_mag'][self.chunk_size : ], dt['phase'][self.chunk_size : ])

        dt['mixed_y'] = torch.multiply(dt['mixed_mag'][self.chunk_size : ], dt['phase'][self.chunk_size : ])
        
        return dt
    
    def cnn2d_recover(self, dt):
        
        dt['pred_y'] = dt['pred_y'].reshape(-1, dt['pred_y'].shape[2])
        
        lens = dt['pred_y'].shape[0]
        
        dt['pred_y'] = torch.multiply(dt['pred_y'], dt['phase'][:lens])
        
        dt['true_y'] = torch.multiply(dt['clean_mag'][:lens], dt['phase'][:lens])
        
        dt['mixed_y'] = torch.multiply(dt['mixed_mag'][:lens], dt['phase'][:lens])
        
        return dt
    
    def forward(self, dt):
        
        if self.target == 'mask':  
            
            dt = self.mask_recover(dt)
        
        if self.transform_type == 'logmag':
            
            for key in ['mixed_mag', 'clean_mag', 'pred_y']:
            
                dt[key] = logmag_transform(dt[key], recover=True)

        elif self.transform_type == 'lps':
            
            for key in ['mixed_mag', 'clean_mag', 'pred_y']:
            
                dt[key] = lps_transform(dt[key], recover=True)
            
        else:
            
            for key in ['mixed_mag', 'clean_mag', 'pred_y']:
                
                dt[key] = quantize_transform(dt[key], recover=True)

        if self.cnn == '1d':

            dt = self.cnn1d_recover(dt)
            
        else:
            
            dt = self.cnn2d_recover(dt)

        for key in ['mixed_y', 'true_y', 'pred_y']:

            dt[key] = torch.istft(dt[key].cpu().detach().T,
                                 n_fft = self.n_fft,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 window=self.window)
            
        return dt            