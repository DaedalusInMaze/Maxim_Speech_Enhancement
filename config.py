DATADIR = "D:/SE"
SAMPLING_RATE = 16000
URL_SPEECH = ["https://www.openslr.org/resources/12/dev-clean.tar.gz"] #this list is to be extended 
SNR = [-10, 5, 0, 5, 10]

N_d = 512  # window duration (samples).
N_s = 128  # window shift (samples).
K = 512  # number of frequency bins.
CHUNK_SIZE = 16 # temporal context 128*512 samples ~ 4sec

EPOCH = 40
BATCH_SIZE = 1
lr = 1e-4

transform_type = 'normal' # option: transform_type = 'logmag', transform_type = 'lps', transform_type = 'normal'
action = 'train' # option: action = 'retrain', action = 'train'
pretrain_model_name = '6_epoch.pth.tar'
stft_type = 'torch' #option: stft_type = 'torch', stft_type = 'librosa'
cnn = '2d'
target = 'mask' #option: target = 'clean_mag', target = 'noise_mag', target = 'mask'