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
from numpy import savetxt

import argparse

noise_type_helper = '0: all, 1: white noise, 2: siren, 3: baby'
parser = argparse.ArgumentParser(description='Speech Enhancement for Hearing Aid Devices')
parser.add_argument('--noise_type', type=int, default=1, help=noise_type_helper)


args = parser.parse_args()

noise_type = {0 : 'white', 1 : 'siren', 2 : 'baby'}


##load data
if args.noise_type:
    training_set_filename = 'dataset-' + noise_type[args.noise_type] + '.pt'
    test_set_filename = 'test-' + noise_type[args.noise_type] + '.pt'
    dataset = torch.load(os.path.join(DATADIR, 'processed', training_set_filename))
    testset = torch.load(os.path.join(DATADIR,'processed', test_set_filename))
else:
    _vars = ['feat_in', 'feat_out', 'angles', 'mask_out', 'noise_out', 'test_feat_in', 'test_feat_in', 'test_feat_out', 'test_angles', 'test_mask_out', 'test_noise_out']
    for _var in _vars:
        exec(f'{_var} = []')
    for i in range(3):
        n = noise_type[i]
        training_set_filename = 'dataset-' + n + '.pt'
        test_set_filename = 'test-' + n + '.pt'
        _in, _out, _ang, _m, _n = torch.load(os.path.join(DATADIR, 'processed', training_set_filename))
        feat_in.append(_in)
        print(len(feat_in))
        feat_out.append(_out)
        angles.append(_ang)
        mask_out.append(_m)
        noise_out.append(_n)
        _in, _out, _ang, _m, _n = torch.load(os.path.join(DATADIR, 'processed', test_set_filename))
        test_feat_in.append(_in)
        test_feat_out.append(_out)
        test_angles.append(_ang)
        test_mask_out.append(_m)
        test_noise_out.append(_n)

        
    dataset = (feat_in, feat_out, angles, mask_out, noise_out)
    testset = (test_feat_in, test_feat_out, test_angles, test_mask_out, test_noise_out)




#load model to cuda
model = SENetv0().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_fn = torch.nn.MSELoss()
num_epochs = 50
training_set = []
test_set = []
#print('preparing training_set')

EXP_NO = 1
metrics = np.zeros((3, len(testset[0]), num_epochs + 1)) #3 metrics for each recording per epoch

for r in range(len(testset[0])):
    noisy_spectrum = testset[0][r]
    clean_spectrum = testset[1][r]
    noisy_angle = testset[2][r] 
    STFT_clean = normalize_quantized_spectrum(clean_spectrum[CHUNK_SIZE:]) * (np.cos(noisy_angle[CHUNK_SIZE:]) + 1j * np.sin(noisy_angle[CHUNK_SIZE:]))
    STFT_noisy = normalize_quantized_spectrum(noisy_spectrum[CHUNK_SIZE:]) * (np.cos(noisy_angle[CHUNK_SIZE:]) + 1j * np.sin(noisy_angle[CHUNK_SIZE:]))
    clean_audio = librosa.istft(STFT_clean.T, hop_length=N_s, win_length=N_d, window='hanning', center=True, dtype=None, length=None)
    noisy_audio = librosa.istft(STFT_noisy.T, hop_length=N_s, win_length=N_d, window='hanning', center=True, dtype=None, length=None)       
    save_wav(os.path.join(DATADIR,'predicted3', 'clean' + str(r) + '.wav'), clean_audio, SAMPLING_RATE) 
    save_wav(os.path.join(DATADIR,'predicted3', 'noisy' + str(r) + '.wav'), noisy_audio, SAMPLING_RATE)
    metrics[0, r, 0] = round(pysepm.SNRseg(clean_audio, noisy_audio, SAMPLING_RATE), 2)
    metrics[1, r, 0] = round(pysepm.stoi.stoi(clean_audio, noisy_audio, SAMPLING_RATE), 2)
    metrics[2, r, 0] = round(pysepm.pesq(clean_audio, noisy_audio, SAMPLING_RATE)[1], 2)
print('training')
for epoch in tqdm(range(num_epochs)):
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
    if True: #epoch and not epoch%2:
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
            STFT_clean = normalize_quantized_spectrum(clean_spectrum[CHUNK_SIZE:]) * (np.cos(noisy_angle[CHUNK_SIZE:]) + 1j * np.sin(noisy_angle[CHUNK_SIZE:]))
            clean_audio = librosa.istft(STFT_clean.T, hop_length=N_s, win_length=N_d, window='hanning', center=True, dtype=None, length=None)
            STFT_predicted = normalize_quantized_spectrum(spectral_subtraction) * (np.cos(noisy_angle[CHUNK_SIZE:]) + 1j * np.sin(noisy_angle[CHUNK_SIZE:]))
            predicted_audio = librosa.istft(STFT_predicted.T, hop_length=N_s, win_length=N_d, window='hanning', center=True, dtype=None, length=None)
            save_wav(os.path.join(DATADIR,'predicted3', 'pred' + str(r) + '_epoch_' + str(epoch + 1) + '.wav'), predicted_audio, SAMPLING_RATE)
            MODEL_PATH = 'models/' + str(EXP_NO) + '_epoch_' + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), MODEL_PATH)

            metrics[0, r, epoch + 1] = round(pysepm.SNRseg(clean_audio, predicted_audio, SAMPLING_RATE), 2)
            metrics[1, r, epoch + 1] = round(pysepm.stoi.stoi(clean_audio, predicted_audio, SAMPLING_RATE), 2)
            metrics[2, r, epoch + 1] = round(pysepm.pesq(clean_audio, predicted_audio, SAMPLING_RATE)[1], 2)

exp_name = str(EXP_NO) + '_SNR.csv'
savetxt(exp_name, metrics[0], fmt='%.2f', delimiter=',')
exp_name = str(EXP_NO) + '_STOI.csv'
savetxt(exp_name, metrics[1], fmt='%.2f', delimiter=',')
exp_name = str(EXP_NO) + '_PESQ.csv'
savetxt(exp_name, metrics[2], fmt='%.2f', delimiter=',')
