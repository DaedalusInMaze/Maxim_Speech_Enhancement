import torch

import torch.nn as nn

import librosa

from utils import quantize_spectrum, normalize_quantized_spectrum

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
                    dt[f'{key}_mag'] = torch.log1p(mag)
                elif self.transform_type == 'lps':
                    dt[f'{key}_mag'] = torch.log10(mag ** 2)
                else:
                    mag = mag / torch.amax(mag)
                    dt[f'{key}_mag'] = quantize_spectrum(mag)

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
            mag = torch.expm1(mag)
            mag = torch.clamp(mag, min= 0, max= torch.amax(mag))
            
        elif self.transform_type == 'lps':
            mag = torch.sqrt(10 ** mag)
        else:
            mag = torch.clamp(mag, min= 0, max= torch.amax(mag))
            mag = normalize_quantized_spectrum(mag)
            
        a = mag.cpu()
        b = torch.cos(phase[self.chunk_size : ]) + 1j * torch.sin(phase[self.chunk_size : ]).cpu()
        # b = torch.exp(1j * phase[self.chunk_size : ])
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
                dt[f'{key}_mag'] = torch.log1p(mag)
            elif self.transform_type == 'lps':
                dt[f'{key}_mag'] = torch.log10(mag ** 2)
            else:
                mag = mag / torch.amax(mag)
                dt[f'{key}_mag'] = quantize_spectrum(mag)

        dt['mask'] = torch.div(dt['clean_mag'], dt['mixed_mag'])
            
        return dt           
    
    

class torch_istft(nn.Module):
    
    def __init__(self, n_fft, hop_length, win_length, chunk_size, device, transform_type='logmag', cnn='1d'):
        
        super(torch_istft, self).__init__()
        
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length= win_length
        self.window = torch.hann_window(n_fft, device= 'cpu')
        self.device = device
        self.transform_type = transform_type
        self.chunk_size = chunk_size
        self.cnn = cnn
    
    def forward(self, dt):
        
        if self.transform_type == 'logmag':
            dt['pred_y'] = torch.expm1(dt['pred_y'])
            dt['pred_y'] = torch.clamp(dt['pred_y'], min= 0, max= torch.amax(dt['pred_y']))
            
        elif self.transform_type == 'lps':
            dt['pred_y'] = torch.sqrt(10 ** dt['pred_y'])
        else:
            dt['pred_y'] = torch.clamp(dt['pred_y'], min= 0, max= torch.amax(dt['pred_y']))
            dt['pred_y'] = normalize_quantized_spectrum(dt['pred_y'])
            
        if self.cnn == '1d':
        
            dt['pred_y'] = torch.multiply(dt['pred_y'], dt['phase'][self.chunk_size : ])

            dt['true_y'] = torch.multiply(dt['clean_mag'][self.chunk_size : ], dt['phase'][self.chunk_size : ])

            dt['mixed_y'] = torch.multiply(dt['mixed_mag'][self.chunk_size : ], dt['phase'][self.chunk_size : ])
        else:
            dt['pred_y'] = dt['pred_y'].reshape(-1, dt['pred_y'].shape[2])
            lens = dt['pred_y'].shape[0]
            dt['pred_y'] = torch.multiply(dt['pred_y'], dt['phase'][:lens])
            dt['true_y'] = torch.multiply(dt['clean_mag'][:lens], dt['phase'][:lens])
            dt['mixed_y'] = torch.multiply(dt['mixed_mag'][:lens], dt['phase'][:lens])

        for key in ['mixed_y', 'true_y', 'pred_y']:

            dt[key] = torch.istft(dt[key].cpu().detach().T,
                                 n_fft = self.n_fft,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 window=self.window)
            
        return dt            