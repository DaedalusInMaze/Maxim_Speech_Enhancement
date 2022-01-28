
############################################### prediction ###########################################

import torch

import torch.nn as nn

import numpy as np

from stft import torch_istft

from config import *

from model_pipeline import SePipline, load_model

from utils import get_audio_path_list, load_audio, save_wav, snr_mixer, generate_test_files

import os

import random
  
model_path = os.path.join(DATADIR, 'models_3')
pretrain_model_name = '15_epoch.pth.tar'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


SE = SePipline(
    n_fft=K, 
    hop_len=N_s, 
    win_len= N_d, 
    window='hanning',
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    transform_type = transform_type)

optimizer = torch.optim.Adam(SE.parameters(), lr=lr)
criterion = nn.MSELoss()

SE= load_model(model= SE, 
              optimizer= optimizer,
              action= 'predict',
              pretrained_model_path = os.path.join(model_path, pretrain_model_name))

test_list = get_audio_path_list(os.path.join(DATADIR, 'valid'), 'flac')

raw_noise_path = os.path.join(DATADIR, 'raw_noise')
noise_path = []
# noise_path.extend(get_audio_path_list(raw_noise_path, 'pt')
noise_path.append(os.path.join(raw_noise_path, 'white_noise.pt'))
noise_path.append(os.path.join(raw_noise_path, 'siren_noise.pt'))
noise_path.append(os.path.join(raw_noise_path, 'baby.pt'))
noise_path.append(os.path.join(raw_noise_path, 'engine_sound.pt'))
noise_path.append(os.path.join(raw_noise_path, 'dog_barking.pt'))
noise_path.append(os.path.join(raw_noise_path, 'traffic_sounds.pt'))
# noise_path.append(os.path.join('testnoise', 'helicopter.pt'))
########### 

noise_path = np.random.choice(noise_path)

test_path = np.random.choice(test_list)

dt = generate_test_files(test_path, noise_path, snr = -10)

for key, value in dt.items():

    dt[key] = torch.tensor(value, device = DEVICE)

with torch.no_grad():

    dt = SE(dt)

iStft = torch_istft(n_fft =K,
                  hop_length=N_s,
                  win_length= N_d,
                  device=DEVICE,
                  chunk_size= CHUNK_SIZE,
                  transform_type =transform_type,
                  cnn=cnn)

dt = iStft(dt)

if not os.path.exists('recovered'):

    os.mkdir('recovered')
    
save_wav(path= os.path.join('recovered', 'mixed_speech.wav'), wav= dt['mixed_y'], fs= SAMPLING_RATE)
save_wav(path= os.path.join('recovered', 'clean_speech.wav'), wav= dt['true_y'], fs= SAMPLING_RATE)
save_wav(path= os.path.join('recovered', 'recovered_speech.wav'), wav= dt['pred_y'], fs= SAMPLING_RATE)

print('The recovered speech is generated!!')