
DATADIR = "SE"
SAMPLING_RATE = 16000
URL_SPEECH = ["https://www.openslr.org/resources/12/dev-clean.tar.gz"] #this list is to be extended 
SNR = 6

N_d = 512  # window duration (samples).
N_s = 128  # window shift (samples).
K = 512  # number of frequency bins.
CHUNK_SIZE = 128 # temporal context 128*512 samples ~ 4sec


