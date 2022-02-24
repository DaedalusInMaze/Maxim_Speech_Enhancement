import os

import torch

import torch.nn as nn

from utils import get_audio_path_list, load_noise, evaluation

from dataset import NoisyData

from torch.utils.data import DataLoader

from config import *

from model_pipeline import SePipline, load_model

from trainer import Trainer

from tqdm import tqdm

CHUNK_SIZE = 128
model_path = os.path.join(DATADIR, 'models_mask_limit10')
pretrain_model_name = 'best_model.pth.tar'
DEVICE = 'cuda'
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

valid_list = get_audio_path_list(os.path.join(DATADIR, 'valid'), 'flac')

raw_noise_path = os.path.join(DATADIR, 'raw_noise')
noise_path = []
noise_path.extend(get_audio_path_list(raw_noise_path, 'pt'))

noises = load_noise(noise_path)

valid_dataset = NoisyData(valid_list, SNR, noises, random_seed=True, random_noise=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
total_loss = 0

total_pesq, total_stoi, total_segsnr = 0, 0, 0
total_pesq2, total_stoi2, total_segsnr2 = 0, 0, 0

valid_bar = tqdm(valid_dataloader)

for batch in valid_bar:

    for key, value in batch.items():

        batch[key] = value.to(DEVICE)

    with torch.no_grad():

        SE.eval()

        batch = SE(batch, train= False)

        loss = criterion(batch['pred_mask'], batch['y'])

        total_loss += loss.item()

        pesq, stoi, segsnr = evaluation(batch['true_y'].numpy(), batch['pred_y'].numpy())

        pesq2, stoi2, segsnr2 = evaluation(batch['true_y'].numpy(), batch['mixed'].squeeze()[:batch['true_y'].shape[0]].cpu().numpy())

        total_pesq += pesq

        total_stoi += stoi

        total_segsnr += segsnr

        total_pesq2 += pesq2

        total_stoi2 += stoi2

        total_segsnr2 += segsnr2

        valid_bar.set_postfix(loss=round(loss.item(),2), pesq=pesq, stoi=stoi, segSNR=segsnr)


lens = len(valid_dataloader)

average_loss = round(total_loss / lens, 4)

average_pesq = round(total_pesq / lens, 2)

average_stoi = round(total_stoi / lens, 2)

average_segsnr = round(total_segsnr / lens, 2)

average_pesq2 = round(total_pesq2 / lens, 2)

average_stoi2 = round(total_stoi2 / lens, 2)

average_segsnr2 = round(total_segsnr2 / lens, 2)

print('\tLoss: ', average_loss, 'pesq: ', average_pesq, 'stoi: ', average_stoi, 'sngSNR: ', average_segsnr)
print('pesq2: ', average_pesq2, 'stoi2: ', average_stoi2, 'sngSNR2: ', average_segsnr2)