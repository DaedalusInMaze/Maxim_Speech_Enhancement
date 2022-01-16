
############################################### prediction ###########################################

import torch, torchaudio

from stft import ISTFT

from config import *

from model import SePipline

from utils import get_audio_path_list, load_audio, save_wav, snr_mixer

import os

import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SE = SePipline(
    n_fft=K, 
    hop_len=N_s, 
    win_len= N_d, 
    window='hanning',
    device=DEVICE,
    chunk_size=CHUNK_SIZE)

SE.load_state_dict(torch.load('models\\model_1.pkl'))

test_list = get_audio_path_list(os.path.join(DATADIR, 'valid'), 'flac')

raw_noise_path = os.path.join(DATADIR, 'raw_noise')

noise_path = os.path.join(raw_noise_path, 'white_noise.pt')

clean = load_audio(path = test_list[51]) ########### HERE predict file

noise = torch.load(noise_path)

noise_start = random.randint(0, noise.shape[0] - clean.shape[0])

noise_snippet = noise[noise_start : noise_start + clean.shape[0]]

mixed, noise = snr_mixer(clean, noise_snippet, SNR)

dt = {'mixed': mixed, 'clean': clean, 'noise': noise}

for key, value in dt.items():

    dt[key] = torch.tensor(value).to(DEVICE).float().unsqueeze(0)

with torch.no_grad():

    dt = SE(dt)

iStft = ISTFT(hop_len=N_s,
              win_len= N_d,
              window='hanning',
              device=DEVICE,
              chunk_size= CHUNK_SIZE)

dt = iStft(dt)

if not os.path.exists('recovered'):

    os.mkdir('recovered')

save_wav(path= os.path.join('recovered', 'clean_speech.wav'), wav= 20 * dt['true_y'], fs= SAMPLING_RATE)
save_wav(path= os.path.join('recovered', 'recovered_speech.wav'), wav=20 * dt['pred_y'], fs= SAMPLING_RATE)

print('The recovered speech is generated!!')