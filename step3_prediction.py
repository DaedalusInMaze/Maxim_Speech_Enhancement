
############################################### prediction ###########################################

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from config import *
from model_pipeline import SePipline, load_model
from stft import torch_istft
from utils import (generate_test_files, get_audio_path_list, load_audio,
                   save_wav, snr_mixer)

model_path = os.path.join(DATADIR, 'models_mask_limit10')
# pretrain_model_name = '12.29_53_epoch.pth.tar'
pretrain_model_name = 'best_model.pth.tar'
DEVICE = 'cpu'
transform_type = 'logmag'

SE = SePipline(
    version='v11',
    n_fft=K, 
    hop_len=N_s, 
    win_len= N_d, 
    window='hanning',
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    transform_type = transform_type,
    target = target,
    cnn = cnn)

optimizer = torch.optim.Adam(SE.parameters(), lr=lr)
criterion = nn.MSELoss()

SE= load_model(model= SE, 
              optimizer= optimizer,
              action= 'predict',
              pretrained_model_path = os.path.join(model_path, pretrain_model_name))

test_list = get_audio_path_list(os.path.join(DATADIR, 'valid'), 'flac')

raw_noise_path = os.path.join(DATADIR, 'raw_noise')
noise_path = []
# noise_path.extend(get_audio_path_list(raw_noise_path, 'pt'))
# noise_path.append(os.path.join(raw_noise_path, 'white_noise.pt'))
# noise_path.append(os.path.join(raw_noise_path, 'siren_noise.pt'))
# noise_path.append(os.path.join(raw_noise_path, 'baby.pt'))
# noise_path.append(os.path.join(raw_noise_path, 'engine_sound.pt'))
noise_path.append(os.path.join(raw_noise_path, 'dog_barking.pt'))
# noise_path.append(os.path.join(raw_noise_path, 'traffic_sounds.pt'))
# noise_path.append(os.path.join('testnoise', 'helicopter.pt'))
########### 

noise_path = np.random.choice(noise_path)

test_path = np.random.choice(test_list)

dt = generate_test_files(test_path, noise_path, snr = 0)

for key, value in dt.items():

    dt[key] = torch.tensor(value, device = DEVICE)

with torch.no_grad():
    torch.cuda.synchronize()
    time_start = time.time()

    dt = SE(dt, train=False)
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)

if not os.path.exists('recovered'):

    os.mkdir('recovered')
    
save_wav(path= os.path.join('recovered', 'mixed_speech.wav'), wav= dt['mixed_y'], fs= SAMPLING_RATE)
save_wav(path= os.path.join('recovered', 'clean_speech.wav'), wav= dt['true_y'], fs= SAMPLING_RATE)
save_wav(path= os.path.join('recovered', 'recovered_speech.wav'), wav= dt['pred_y'], fs= SAMPLING_RATE)

print('The recovered speech is generated!!')
