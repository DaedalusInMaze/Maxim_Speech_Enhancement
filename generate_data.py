from utils import *

import random

import torch

from config import *


# NOTE: for quick testing to only download small portion of repo, make 100 to resample all repo (VERY LONG)
speech_repo_resample_percent = 2  # 1-100%
noise_repo_resample_percent = 2

#create paths
raw_folder_speech = os.path.join('SE', 'raw_speech')
raw_folder_noise = os.path.join('SE', 'raw_noise')
processed_folder = os.path.join('SE', 'processed')

resampled_speech_file = "speech_" + str(SAMPLING_RATE / 1000) + "KHz.pt"
resampled_noise_file = "noise_" + str(SAMPLING_RATE / 1000) + "KHz.pt"
data_file = 'dataset-speech.pt'

print(f'\rWarning: Resample {speech_repo_resample_percent}% of data repo and {noise_repo_resample_percent}% of noise repo')

makedir_exist_ok(raw_folder_speech)
makedir_exist_ok(raw_folder_noise)
makedir_exist_ok(processed_folder)

filename_speech = URL_SPEECH.rpartition('/')[2]
filename_noise = URL_NOISE.rpartition('/')[2].rsplit('?')[0]


## download and extract data
download_and_extract_archive(URL_SPEECH, download_root = raw_folder_speech, filename = filename_speech)
download_and_extract_archive(URL_NOISE, download_root = raw_folder_noise, filename = filename_noise)


## if resampled clean speech data does not exist then create one ( torch file )
## if it exist, load data( torch file ) to `data` variable
if not os.path.exists(os.path.join(processed_folder, resampled_speech_file)):
    print(f'\rResampling data @ {SAMPLING_RATE}Hz')
    DATA = resample(raw_folder_speech, sr = SAMPLING_RATE, ext = '.flac', max_prog = speech_repo_resample_percent)

    # save resampled speech
    print(f'\rSaving resampled data: {os.path.join(processed_folder, resampled_speech_file)}')
    torch.save(DATA, os.path.join(processed_folder, resampled_speech_file))
else:
    print(f'\rWarning: Resampled data file exists (remove and run again to regenerate), start loading...')
    DATA = torch.load(os.path.join(processed_folder, resampled_speech_file))


## if resampled noise data does not exist then create one ( torch file )
## if it exist, load data( torch file ) to `noise` variable
if not os.path.exists(os.path.join(processed_folder, resampled_noise_file)):
    print(f'\rResampling noise @ {SAMPLING_RATE}Hz\r')
    NOISE = resample(raw_folder_noise, sr = SAMPLING_RATE, ext = '.wav', max_prog = noise_repo_resample_percent)

    # save resampled noise
    print(f'\rSaving resampled noise: {os.path.join(processed_folder, resampled_noise_file)}')
    torch.save(NOISE, os.path.join(processed_folder, resampled_noise_file))
else:
    print(f'\rWarning: Resampled noise file exists (remove and run again to regenerate), start loading...')
    NOISE = torch.load(os.path.join(processed_folder, resampled_noise_file))




noise_in = []

for noise_frame in NOISE:
    noise_in = np.append(noise_in, noise_frame, axis = 0)  # make an nx1 noise array


for count, data_frame in tqdm(enumerate(DATA)):
    print(f'\rProcessing {count + 1} of {len(DATA)} data', end = "")
    noise_start = random.randint(0, noise_in.shape[0]- data_frame.shape[0])
    noise = noise_in[noise_start: noise_start + data_frame.shape[0]]
    noisy_frame, valid, scaled_clean, scaled_noise = snr_mixer(data_frame, noise, snr)
    noisy_feat = np.abs(librosa.stft(noisy_frame, n_fft=fft_num, win_length=WIN_LEN, hop_length=WIN_HOP, window='hanning').T)
    noisy_angle = np.angle(librosa.stft(noisy_frame, n_fft=fft_num, win_length=WIN_LEN, hop_length=WIN_HOP, window='hanning').T)
    clean_feat = np.abs(librosa.stft(scaled_clean, n_fft=fft_num, win_length=WIN_LEN, hop_length=WIN_HOP, window='hanning').T)
    # normalize to 0-1.0
    noisy_feat = noisy_feat / np.amax(noisy_feat)
    clean_feat = clean_feat / np.amax(clean_feat)
    
    chunk_start = 0
    clean_start = chunk_size - 1
    while True:
        if chunk_start + chunk_size > noisy_feat.shape[0]:
            break

        feat_in = noisy_feat[chunk_start: chunk_start + chunk_size, :]
        angles = noisy_angle[chunk_start: chunk_start + chunk_size, :]
        feat_out = clean_feat[clean_start, :]
        chunk_start += 1
        clean_start += 1
        # only first column?
        feat_in = np.expand_dims(feat_in, axis=0)
        feat_out = np.expand_dims(feat_out, axis=0)
        angles = np.expand_dims(angles, axis=0)

        
        if flag:
            stft_in = feat_in.copy()
            label = feat_out.copy()
            angle = angles.copy()
            flag = False
        else:
            stft_in = np.concatenate((stft_in, feat_in), axis=0)
            label = np.concatenate((label, feat_out), axis = 0)
            angle = np.concatenate((angle, angles), axis = 0)
        #print(stft_in.shape)
        #print(label.shape)
    if count == 1:
        break

    
print('\rQuantizing stft output and labels...\r')
stft_in_q = quantize_audio(stft_in)
label_q = quantize_audio(label)
# 0 (90%): train and test   1 (10%): validate
data_type = (np.random.rand(label_q.shape[0], 1) + 0.1).astype(int)
print(f'\rtrain data: {stft_in_q.shape}  labels: {label_q.shape}  data type:{data_type.shape}')
speech_dataset = (stft_in_q, label_q, angle, data_type)
torch.save(speech_dataset, os.path.join(processed_folder, data_file))
print(f'\rDataset created: {os.path.join(processed_folder, data_file)}')