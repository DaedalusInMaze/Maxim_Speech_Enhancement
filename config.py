DATADIR = "D:/SE"
SAMPLING_RATE = 16000
URL_SPEECH = ["https://www.openslr.org/resources/12/dev-clean.tar.gz"] #this list is to be extended 
SNR = [-10, 5, 0, 5, 10]

N_d = 512  # window duration (samples).
N_s = 128  # window shift (samples).
K = 512  # number of frequency bins.
CHUNK_SIZE = 128 # temporal context 128*512 samples ~ 4sec

EPOCH = 40
BATCH_SIZE = 1
lr = 1e-4

transform_type = 'logmag' # option: transform_type = 'logmag', transform_type = 'lps', transform_type = 'normal'
action = 'retrain' # option: action = 'retrain', action = 'train'
pretrain_model_name = 'best_model.pth.tar'

stft_type = 'torch' #option: stft_type = 'torch', stft_type = 'librosa'
cnn = '2d'
target = 'clean_mag' #option: target = 'clean_mag', target = 'noise_mag', target = 'mask'
version = 'v4'