import torch 

from torch.utils.data import Dataset

from utils import load_audio, snr_mixer

import numpy as np

from tqdm import tqdm

class NoisyData(Dataset):

    def __init__(self, audio_list,  snr, noise, random_seed = None, random_noise= False, ):

        super(NoisyData, self).__init__()

        self.audios = audio_list

        self.snr = snr
        
        self.noise = noise
        
        self.random_noise = random_noise
        
        self.random_seed = random_seed
        
    
    def __len__(self):

        return len(self.audios)
    
    def __getitem__(self, idx):
        
        if self.random_seed:
            
            np.random.seed(idx)
        
        clean = load_audio(path = self.audios[idx])
        
        if self.random_noise:
            
            noise_type = np.random.randint(0, len(self.noise))
            
            noise_snr = np.random.choice(self.snr)
            
        else:
            
            noise_type = 0 #baby crying
            
            noise_snr = 6
        
        while True:
            noise_start = np.random.randint(0, self.noise[noise_type].shape[0] - clean.shape[0] + 1)

            noise_snippet = self.noise[noise_type][noise_start : noise_start + clean.shape[0]]

            valid, mixed, noise = snr_mixer(clean, noise_snippet, noise_snr)
            
            if valid:
                break
        
        dt = {'mixed': mixed, 'clean': clean, 'noise': noise}

        return dt
        

