################################## data flow #######################

import os

import torch

import torch.nn as nn

from utils import get_audio_path_list

from dataset import NoisyData

from torch.utils.data import DataLoader

from config import *

from model import SePipline

from trainer import Trainer


recovered_path = os.path.join(DATADIR, 'recovered')
model_path = os.path.join(DATADIR, 'models')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_list = get_audio_path_list(os.path.join(DATADIR, 'train'), 'flac')
valid_list = get_audio_path_list(os.path.join(DATADIR, 'valid'), 'flac')
test_list = get_audio_path_list('testaudio', 'flac')

raw_noise_path = os.path.join(DATADIR, 'raw_noise')
noise_path = []
noise_path.append(os.path.join(raw_noise_path, 'white_noise.pt'))
noise_path.append(os.path.join(raw_noise_path, 'siren_noise.pt'))
noise_path.append(os.path.join(raw_noise_path, 'baby.pt'))

noise_dataset = NoisyData(train_list[:], SNR, noise_path, random_noise=True)
valid_dataset = NoisyData(valid_list[:50], SNR, noise_path, random_seed=True)
test_dataset = NoisyData(test_list, SNR, noise_path, random_seed=True)

train_dataloader = DataLoader(noise_dataset, batch_size=BATCH_SIZE, shuffle=True) 
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

##########模块化后##############
SE = SePipline(
    n_fft=K, 
    hop_len=N_s, 
    win_len= N_d, 
    window='hanning',
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    transform_type = transform_type)
##########

optimizer = torch.optim.Adam(SE.parameters(), lr=lr)
criterion = nn.MSELoss()

trainer = Trainer(model= SE,
                  train_loader= train_dataloader, 
                  valid_loader= valid_dataloader, 
                  test_loader= test_dataloader,
                  optimizer= optimizer, 
                  criterion= criterion, 
                  device= DEVICE)

trainer.train(epochs= EPOCH,
              save_model=True,
              hop_len=N_s, 
              win_len= N_d, 
              window='hanning',
              chunk_size=CHUNK_SIZE,
              n_fft = K,
              recovered_path = recovered_path,
              fs = SAMPLING_RATE,
              transform_type = transform_type,
              model_path = model_path)

print('training finished')