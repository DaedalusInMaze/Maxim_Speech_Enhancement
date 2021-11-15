from utils import *
import random
import torch
from config import *
import tensorflow as tf

import argparse


parser = argparse.ArgumentParser(description='Speech Enhancement for Hearing Aid Devices')
parser.add_argument('--noise_type', type=int, default=1, help='0: white noise, 1: siren')
parser.add_argument('--clean_file_no', type=int, default=0, help='select file source')

args = parser.parse_args()

noise_type = {0 : 'white', 1 : 'siren'}

#create paths
raw_folder_speech = os.path.join(DATADIR, 'raw_speech')
raw_folder_noise = os.path.join(DATADIR, 'raw_noise')
processed_folder = os.path.join(DATADIR, 'processed')
makedir_exist_ok(processed_folder)
speech_file = os.path.join(raw_folder_speech, "speech_" + str(args.clean_file_no) + ".pt")
noise_file = os.path.join(raw_folder_noise, noise_type[args.noise_type] + '_noise.pt')
data_file = 'dataset-speech.pt'

if os.path.exists(os.path.join(raw_folder_speech, speech_file)):
	clean = torch.load(os.path.join(raw_folder_speech, speech_file))

if os.path.exists(os.path.join(raw_folder_noise, noise_file)):
	noise = torch.load(os.path.join(raw_folder_noise, noise_file))

#first_chunk = True
feat_in = []
angles = []
feat_out = []
mask_out = []
data_types = []
for count, data_frame in tqdm(enumerate(clean)):
	print(f'\rProcessing {count + 1} of {len(clean)} data', end = "")
	noise_start = random.randint(0, noise.shape[0]- data_frame.shape[0])
	noise_snippet = noise[noise_start: noise_start + data_frame.shape[0]]
	noisy_frame, valid, scaled_clean, scaled_noise = snr_mixer(data_frame, noise_snippet, SNR)
	#noisy_feat = tf.abs(tf.signal.stft(noisy_frame, N_d, N_s, K, window_fn=W, pad_end=True))
	#noisy_angle = tf.math.angle(tf.signal.stft(noisy_frame, N_d, N_s, K, window_fn=W, pad_end=True))
	#clean_feat = tf.abs(tf.signal.stft(data_frame, N_d, N_s, K, window_fn=W, pad_end=True))

	noisy_feat = np.abs(librosa.stft(noisy_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
	noisy_angle = np.angle(librosa.stft(noisy_frame, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
	clean_feat = np.abs(librosa.stft(scaled_clean, n_fft=K, win_length=N_d, hop_length=N_s, window='hanning').T)
	# normalize to 0-1.0
	noisy_feat = noisy_feat / np.amax(noisy_feat)
	clean_feat = clean_feat / np.amax(clean_feat)
	mask =  tf.truediv(clean_feat, noisy_feat)
	feat_in.append(quantize_spectrum(noisy_feat))
	feat_out.append(quantize_spectrum(clean_feat))
	data_type = (np.random.rand(clean_feat.shape[0], 1) + 0.1).astype(int)
	data_types.append(data_type)
	angles.append(noisy_angle)
	mask_out.append(quantize_spectrum(mask))

speech_dataset = (feat_in, feat_out, angles, mask_out, data_types)
torch.save(speech_dataset, os.path.join(processed_folder, data_file))
print(f'\rDataset created: {os.path.join(processed_folder, data_file)}')