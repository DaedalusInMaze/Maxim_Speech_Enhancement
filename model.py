import torch.nn as nn

import torch

from stft import STFT, torch_stft

from sliding_window import ChunkData

from model_senet import SENetv2



class SePipline(nn.Module):
    def __init__(self, n_fft, hop_len, win_len, window, device, chunk_size, transform_type='logmag'):

        super(SePipline, self).__init__()

        self.model = nn.Sequential(
            # STFT(n_fft=n_fft, hop_len=hop_len, win_len= win_len, window=window),
            torch_stft(n_fft=n_fft, hop_length=hop_len, win_length= win_len, device = device, transform_type= transform_type),
            ChunkData(chunk_size= chunk_size),
            SENetv2()
        ).to(device)

    def forward(self, dt):

        dt = self.model(dt)

        return dt


def load_model(model, optimizer, action='train', **kargs):
    
    if action == 'train':
        
        print('train from begin')
    
        epoch = 1
        
        return epoch, model, optimizer
    
    elif action == 'retrain':
        
        print(f"load model from {kargs['pretrained_model_path']}")
        
        checkpoint = torch.load(kargs['pretrained_model_path'])
        
        epoch = checkpoint['epoch'] + 1
        
        model.eval()
        
        model.load_state_dict(checkpoint['model'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for state in optimizer.state.values():
            
            for k, v in state.items():
                
                if isinstance(v, torch.Tensor):
                    
                    state[k] = v.cuda()
        
        return epoch, model, optimizer