from utils import *
import random
import torch
from config import *

import argparse


parser = argparse.ArgumentParser(description='Speech Enhancement for Hearing Aid Devices')
parser.add_argument('--noise_type', type=int, default=0, help='0: white noise, 1: siren, 2: baby')
parser.add_argument('--clean_file_no', type=int, default=0, help='select file source')

args = parser.parse_args()

noise_type = {0 : 'white', 1 : 'siren', 2 : 'baby'}
# NOTE: for quick testing to only download small portion of repo, make 100 to resample all repo (VERY LONG)
speech_repo_resample_percent = 100  # 1-100%

#create paths
raw_folder_speech = os.path.join(DATADIR, 'raw_speech')
raw_folder_noise = os.path.join(DATADIR, 'raw_noise')

resampled_speech_file = 'speech_' + str(args.clean_file_no) + '.pt'
resampled_noise_file = noise_type[args.noise_type] + '_noise.pt'

print(f'\rWarning: Resample {speech_repo_resample_percent}% of data repo')
makedir_exist_ok(raw_folder_speech)
filename_speech = URL_SPEECH[args.clean_file_no].rpartition('/')[2]
## download and extract data
download_and_extract_archive(URL_SPEECH[args.clean_file_no], download_root = raw_folder_speech, filename = filename_speech)

print(f'\rResampling data @ {SAMPLING_RATE}Hz')
DATA = resample(raw_folder_speech, sr = SAMPLING_RATE, ext = '.flac', max_prog = speech_repo_resample_percent)

# save resampled speech
print(f'\rSaving resampled clean data: {os.path.join(raw_folder_speech, resampled_speech_file)}')
torch.save(DATA, os.path.join(raw_folder_speech, resampled_speech_file))

print(f'\rSaving resampled noise data: {os.path.join(raw_folder_noise, resampled_noise_file)}')
DATA, samplerate = librosa.load(os.path.join(raw_folder_noise, noise_type[args.noise_type] + '.wav'), sr=SAMPLING_RATE)
torch.save(DATA, os.path.join(raw_folder_noise, resampled_noise_file))