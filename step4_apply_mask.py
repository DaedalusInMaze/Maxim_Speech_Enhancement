import torch
import os
import tensorflow as tf
import librosa
import numpy as np
from config import *
from tqdm import tqdm
from utils import save_wav, normalize_quantized_spectrum

##load data
dataset = torch.load(os.path.join(DATADIR,'processed/dataset-speech.pt'))

for record in tqdm(range(len(dataset[0]))):
    noisy_spectrum = dataset[0][record]
    clean_spectrum = dataset[1][record]
    noisy_angle = dataset[2][record]
    #mask = dataset[3][record]
    #creating audio from the STFTs
    STFT_noisy = normalize_quantized_spectrum(noisy_spectrum) * (np.cos(noisy_angle) + 1j * np.sin(noisy_angle))
    # The following one is to be predicted using the CNN
    STFT_clean = normalize_quantized_spectrum(clean_spectrum) * (np.cos(noisy_angle) + 1j * np.sin(noisy_angle))
    noisy_audio = librosa.istft(STFT_noisy.T, hop_length=N_s, win_length=N_d, window='hamming', center=True, dtype=None, length=None)
    clean_audio = librosa.istft(STFT_clean.T, hop_length=N_s, win_length=N_d, window='hamming', center=True, dtype=None, length=None)
    save_wav(os.path.join(DATADIR,'processed', 'noisy' + str(record) + '.wav'), noisy_audio, SAMPLING_RATE)
    save_wav(os.path.join(DATADIR,'processed', 'clean' + str(record) + '.wav'), clean_audio, SAMPLING_RATE)


