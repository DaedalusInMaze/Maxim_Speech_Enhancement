import torch
import librosa
import numpy as np
from config import *
from simple_model import SENetv0
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize
from utils import save_wav, normalize_quantized_spectrum
import pysepm

##load data
dataset = torch.load(os.path.join(DATADIR,'processed/dataset-speech.pt'))
testset = torch.load(os.path.join(DATADIR,'processed/test-speech.pt'))

#load model to cuda
model = SENetv0().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_fn = torch.nn.MSELoss()

training_set = []
test_set = []
#print('preparing training_set')

EXP_NO = 1
print('training')
for epoch in tqdm(range(3)):
    _loss = 0
    for record in range(len(dataset[0])):
        noisy_spectrum = dataset[0][record]
        clean_spectrum = dataset[1][record]
        noisy_angle = dataset[2][record]
        noise_spectrum = dataset[4][record]
        #datatype = dataset[5][record]
        feat_in = np.zeros((noisy_spectrum.shape[0] - CHUNK_SIZE , CHUNK_SIZE, noisy_spectrum.shape[1]))
        feat_out = np.zeros((noisy_spectrum.shape[0] - CHUNK_SIZE, noisy_spectrum.shape[1]))
        for frame in range(noisy_spectrum.shape[0] - CHUNK_SIZE):
            feat_in[frame, :, :] = noisy_spectrum[frame : frame + CHUNK_SIZE, :]
            feat_out[frame, :] = noise_spectrum[CHUNK_SIZE + frame, :] 
        au_in = torch.from_numpy(feat_in).permute(0, 2, 1).cuda().float()
        au_out = torch.from_numpy(feat_out).cuda().float()
        pred_out = torch.squeeze(model(au_in))
        loss = loss_fn(au_out, pred_out)
        _loss += loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(_loss)
    if epoch and not epoch%2:
        for r in range(len(testset[0])):
            noisy_spectrum = testset[0][r]
            clean_spectrum = testset[1][r]
            noisy_angle = testset[2][r]
            noise_spectrum = testset[4][r] 
            feat_in = np.zeros((noisy_spectrum.shape[0] - CHUNK_SIZE , CHUNK_SIZE, noisy_spectrum.shape[1]))
            feat_out = np.zeros((noisy_spectrum.shape[0] - CHUNK_SIZE, noisy_spectrum.shape[1]))
            for frame in range(noisy_spectrum.shape[0] - CHUNK_SIZE):
                feat_in[frame, :, :] = noisy_spectrum[frame : frame + CHUNK_SIZE, :]
                feat_out[frame, :] = clean_spectrum[CHUNK_SIZE + frame, :]
            au_in = torch.from_numpy(feat_in).permute(0, 2, 1).cuda().float()
            pred_out = torch.squeeze(model(au_in)).cpu().detach().numpy()
            spectral_subtraction = noisy_spectrum[CHUNK_SIZE:] - pred_out
            spectral_subtraction = np.clip(spectral_subtraction, 0, np.amax(spectral_subtraction))

            STFT_noisy = normalize_quantized_spectrum(noisy_spectrum[CHUNK_SIZE:]) * (np.cos(noisy_angle[CHUNK_SIZE:]) + 1j * np.sin(noisy_angle[CHUNK_SIZE:]))
            STFT_predicted = normalize_quantized_spectrum(spectral_subtraction) * (np.cos(noisy_angle[CHUNK_SIZE:]) + 1j * np.sin(noisy_angle[CHUNK_SIZE:]))
            STFT_clean = normalize_quantized_spectrum(feat_out) * (np.cos(noisy_angle[CHUNK_SIZE:]) + 1j * np.sin(noisy_angle[CHUNK_SIZE:]))
            noisy_audio = librosa.istft(STFT_noisy.T, hop_length=N_s, win_length=N_d, window='hanning', center=True, dtype=None, length=None)
            predicted_audio = librosa.istft(STFT_predicted.T, hop_length=N_s, win_length=N_d, window='hanning', center=True, dtype=None, length=None)
            print(noisy_audio.shape)
            clean_audio = librosa.istft(STFT_clean.T, hop_length=N_s, win_length=N_d, window='hanning', center=True, dtype=None, length=None)
            
            save_wav(os.path.join(DATADIR,'predicted3', 'noisy' + str(r) + '.wav'), noisy_audio, SAMPLING_RATE)
            save_wav(os.path.join(DATADIR,'predicted3', 'pred' + str(r) + '_epoch_' + str(epoch) + '.wav'), predicted_audio, SAMPLING_RATE)
            save_wav(os.path.join(DATADIR,'predicted3', 'noise' + str(r) + '_epoch_' + str(epoch) + '.wav'), clean_audio, SAMPLING_RATE)
            MODEL_PATH = 'models/' + str(EXP_NO) + '_epoch_' + str(epoch) + '.pkl'
            torch.save(model.state_dict(), MODEL_PATH)
            print('seg SNR')
            print(round(pysepm.SNRseg(clean_audio, noisy_audio, SAMPLING_RATE), 2))
            print(round(pysepm.SNRseg(clean_audio, predicted_audio, SAMPLING_RATE), 2))
            print(round(pysepm.SNRseg(clean_audio, clean_audio, SAMPLING_RATE), 2))

            print('STOI')
            print(pysepm.stoi.stoi(clean_audio, noisy_audio, SAMPLING_RATE))
            print(pysepm.stoi.stoi(clean_audio, predicted_audio, SAMPLING_RATE))

            print('PESQ')
            print(pysepm.pesq(clean_audio, noisy_audio, SAMPLING_RATE))
            print(pysepm.pesq(clean_audio, predicted_audio, SAMPLING_RATE))
