import torch
import torch.nn as nn

from senet import *
from sliding_window import *
from stft import *


class SePipline(nn.Module):
    def __init__(self, version, n_fft, hop_len, win_len, window, device, chunk_size, transform_type='logmag', stft_type='torch', **kwargs):

        super(SePipline, self).__init__()

        if version == 'v1':
            print(version)
            chunk = ChunkData(chunk_size= 128, target= 'clean_mag')
            model = SENetv1()
        elif version == 'v2':
            print(version)
            chunk = ChunkDatav2(chunk_size= 16, target= 'clean_mag')
            model = SENetv3()
        elif version == 'v3':
            print(version) #models_mask
            chunk = ChunkDatav2(chunk_size= 16, target= 'mask')
            model = SENetv3()
        elif version == 'v4':
            print(version)
            chunk = ChunkDatav2(chunk_size= chunk_size, target= 'clean_mag')
            model = SENetv4(64)
        elif version == 'v5':
            print(version)
            chunk = ChunkDatav2(chunk_size= chunk_size, target= 'mask')
            model = SENetv4(64)
        elif version == 'v6':
            print(version)
            chunk = ChunkDatav2(chunk_size= chunk_size, target= 'mask')
            model = SENetv5(16)
        elif version == 'v7':
            print(version)
            chunk = ChunkDatav2(chunk_size= chunk_size, target= 'mask')
            model = SENetv5(16)
        elif version == 'v8':
            print(version)
            chunk = ChunkDatav3(chunk_size= chunk_size, target= 'mask')
            model = SENetv7()
        elif version == 'v9':#models_mask_limit4 # chunk_size=16
            print(version)
            chunk = ChunkDatav3(chunk_size= chunk_size, target= 'mask')
            model = SENetv8()
        elif version == 'v10':#models_mag_limit2 # chunk_size=16
            print(version)
            chunk = ChunkDatav3(chunk_size= chunk_size, target= 'clean_mag')
            model = SENetv8()
        elif version == 'v11':#models_mask_limit5 # chunk_size=32 and models_mask_limit6 # chunk_size=128 and # models_mask_limit7 #chunk_size = 64 and #models_mask_limit8 # chunk_size=16 #models_mask_limit9 chunk_size=128 with snr_mixer2, #models_mask_limit10 chunk_size=128 with snr_mixer2 and fixed small bug
            print(version)
            chunk = ChunkDatav3(chunk_size= chunk_size, target= 'mask')
            model = SENetv10()
        elif version == 'v12':
            print(version) #models_mask2
            chunk = ChunkDatav2(chunk_size= 16, target= 'mask')
            model = SENetv11()

        
        if stft_type == 'torch':
            _stft = torch_stft(n_fft=n_fft, hop_length=hop_len, win_length= win_len, device = device, transform_type= transform_type)
            _istft = torch_istft(n_fft =n_fft, hop_length=hop_len, win_length= win_len, device=device, chunk_size= chunk_size, transform_type =transform_type, target= kwargs['target'], cnn = kwargs['cnn'])
            
        elif stft_type == 'librosa':
            _stft = STFT(n_fft=n_fft, hop_len=hop_len, win_len= win_len, window=window, transform_type= transform_type)
            _istft = ISTFT(hop_len=hop_len, win_len = win_len, window= window, device= device, chunk_size= chunk_size, transform_type= transform_type)
            
        # self.model = nn.Sequential(
        #     _stft,
        #     chunk,
        #     model,
        #     _istft
        # ).to(device)
        
        self.stft = _stft.to(device)
        self.chunk = chunk.to(device)
        self.model = model.to(device)
        self.istft = _istft.to(device)

    def forward(self, dt, train=True):

        dt = self.stft(dt)
        dt = self.chunk(dt)
        dt = self.model(dt)
        if not train:
            dt = self.istft(dt)

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
    
    elif action == 'predict':
        
        print(f"load model from {kargs['pretrained_model_path']}")
        
        checkpoint = torch.load(kargs['pretrained_model_path'])  
        
        model.eval()
        
        model.load_state_dict(checkpoint['model'])
        
        return model
