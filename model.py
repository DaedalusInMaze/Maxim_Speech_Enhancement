import torch.nn as nn

import torch

from stft import STFT, torch_stft

from sliding_window import ChunkData

from model_senet import SENetv1

class SENetv0(nn.Module):

    # output = 161 STFT feats
    def __init__(self, num_channels=257, dimensions=(257, 1), bias=True, **kwargs):
        super().__init__() ## 继承nn.Module的属性

        self.project = nn.Sequential(

            nn.Conv1d(257, 256, 9), # in : 161 x 128; out: 256 x 120
            nn.ReLU(inplace=True),   
            
            nn.MaxPool1d(2),# in : 256 x 120; out: 256 x 60           
            nn.Conv1d(256, 128, 9),# in : 256 x 60; out: 128 x 52
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            #########################################################################
            nn.Conv1d(128, 128, 9, padding=4),# in : 128 x 52; out: 128 x 52
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 9, padding=4),# in : 128 x 52; out: 128 x 52
            nn.ReLU(),
            nn.BatchNorm1d(128),
            #########################################################################
            nn.MaxPool1d(2),# in : 128 x 52; out: 128 x 26
            
            nn.Conv1d(128, 96, 9),# in : 128 x 26, out: 96 x 18
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.MaxPool1d(2), # in : 96 x 18, out: 96 x 9            
            nn.Conv1d(96, 257, 9), # in : 96 x 9, out: 161 x 1
            nn.ReLU()
        )

    def forward(self, dt):
        x = dt['x']  # in: (batch, frequency, time), out:(batch, frequency, time)
        x = self.project(x) # in:(batch_size, 257, 128) out:(batch_size, 257, 1)
        dt['pred_y'] = torch.squeeze(x)
        return dt

class SePipline(nn.Module):
    def __init__(self, n_fft, hop_len, win_len, window, device, chunk_size, transform_type='logmag'):

        super(SePipline, self).__init__()

        self.model = nn.Sequential(
            # STFT(n_fft=n_fft, hop_len=hop_len, win_len= win_len, window=window),
            torch_stft(n_fft=n_fft, hop_length=hop_len, win_length= win_len, device = device, transform_type= transform_type),
            ChunkData(chunk_size= chunk_size),
            SENetv1()
        ).to(device)

    def forward(self, dt):

        dt = self.model(dt)

        return dt



