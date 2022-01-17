from utils import *
import random
import torch
from config import *
import tensorflow as tf
import gc

import argparse

noise_type_helper = '0: all, 1: white, 2: siren, 3 : baby'
parser = argparse.ArgumentParser(description='Speech Enhancement for Hearing Aid Devices')
parser.add_argument('--noise_type', type=int, default=0, help=noise_type_helper)
parser.add_argument('--clean_file_no', type=int, default=0, help='select file source')

args = parser.parse_args()

noise_type = {1 : 'white', 2 : 'siren', 3 : 'baby'}

#create paths
raw_folder_speech = os.path.join(DATADIR, 'raw_speech')
raw_folder_noise = os.path.join(DATADIR, 'raw_noise')
processed_folder = os.path.join(DATADIR, 'processed')
makedir_exist_ok(processed_folder)
speech_file = os.path.join(raw_folder_speech, "speech_" + str(args.clean_file_no) + ".pt")
if args.noise_type:
    print(f'Creating the dataset for the {noise_type[args.noise_type]} noise')
    noise_file = os.path.join(raw_folder_noise, noise_type[args.noise_type] + '_noise.pt')
    if os.path.exists(noise_file):
        noise_set = [torch.load(noise_file)]
else:
    print(f'Creating the dataset for all noise types combined')
    noise_set = []
    for f in os.listdir(raw_folder_noise):
        if f.endswith('.pt'):
            noise_file = os.path.join(raw_folder_noise, f)
            noise_set.append(torch.load(noise_file))
    print(len(noise_set))

test_file = os.path.join(raw_folder_speech, 'test.pt')
data_file = 'dataset-speech.pt'
if os.path.exists(speech_file):
	clean = torch.load(speech_file)
    
if os.path.exists(test_file):
	test = torch.load(test_file)


feat_in = []
angles = []
feat_out = []
noise_out = []
mask_out = []
for count, data_frame in tqdm(enumerate(clean)):
    for noise in noise_set:
        noise_start = random.randint(0, noise.shape[0]- data_frame.shape[0])
        noise_snippet = noise[noise_start: noise_start + data_frame.shape[0]]
        noisy_frame, valid, scaled_noise = snr_mixer(data_frame, noise_snippet, SNR)
        if valid:
            noisy_feat = np.abs(librosa.stft(noisy_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            noisy_angle = np.angle(librosa.stft(noisy_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            clean_feat = np.abs(librosa.stft(data_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            noise_feat = np.abs(librosa.stft(scaled_noise, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            # normalize to 0-1.0
            noisy_feat = noisy_feat / np.amax(noisy_feat)
            clean_feat = clean_feat / np.amax(clean_feat)
            noise_feat = noise_feat / np.amax(noise_feat)
            mask =  tf.truediv(clean_feat, noisy_feat)
            feat_in.append(quantize_spectrum(noisy_feat))
            feat_out.append(quantize_spectrum(clean_feat))
            noise_out.append(quantize_spectrum(noise_feat))
            # is the recoding used in training (0) or test (1)
            angles.append(noisy_angle)
            mask_out.append(quantize_spectrum(mask))
print(len(feat_in))

speech_dataset = (feat_in, feat_out, angles, mask_out, noise_out)
print('reached here')
to_delete = ['feat_in', 'feat_out', 'angles', 'mask_out', 'noise_out']
for _var in to_delete:
    if _var in locals() or _var in globals():
        exec(f'del {_var}')
gc.collect()
torch.save(speech_dataset, os.path.join(processed_folder, data_file))
print(f'\rDataset created: {os.path.join(processed_folder, data_file)}')

test_feat_in = []
test_angles = []
test_feat_out = []
test_mask_out = []
test_noise_out = []


for count, data_frame in tqdm(enumerate(test)):
    for noise in noise_set:
        noise_start = random.randint(0, noise.shape[0]- data_frame.shape[0])
        noise_snippet = noise[noise_start: noise_start + data_frame.shape[0]]
        noisy_frame, valid, scaled_noise = snr_mixer(data_frame, noise_snippet, SNR)
        if valid:
            noisy_feat = np.abs(librosa.stft(noisy_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            noisy_angle = np.angle(librosa.stft(noisy_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            clean_feat = np.abs(librosa.stft(data_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            noise_feat = np.abs(librosa.stft(scaled_noise, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
            # normalize to 0-1.0
            noisy_feat = noisy_feat / np.amax(noisy_feat)
            clean_feat = clean_feat / np.amax(clean_feat)
            noise_feat = noise_feat / np.amax(noise_feat)
            mask =  tf.truediv(clean_feat, noisy_feat)
            test_feat_in.append(quantize_spectrum(noisy_feat))
            test_feat_out.append(quantize_spectrum(clean_feat))
            test_noise_out.append(quantize_spectrum(noise_feat))
            test_angles.append(noisy_angle)
            test_mask_out.append(quantize_spectrum(mask))

test_dataset = (test_feat_in, test_feat_out, test_angles, test_mask_out, test_noise_out)
torch.save(test_dataset, os.path.join(processed_folder, 'test-speech.pt'))


