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
action = 'train' # option: action = 'retrain', action = 'train'
pretrain_model_name = '31_epoch.pth.tar'
stft_type = 'librosa' #option: stft_type = 'torch', stft_type = 'librosa'