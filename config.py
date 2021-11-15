import functools
from tensorflow.python.ops.signal import window_ops

DATADIR = "D:/SE"
SAMPLING_RATE = 16000
URL_SPEECH = ["https://www.openslr.org/resources/12/dev-clean.tar.gz"] #this list is to be extended 
SNR = 6

N_d = 512  # window duration (samples).
N_s = 128  # window shift (samples).
K = 512  # number of frequency bins.
W = functools.partial(window_ops.hamming_window, periodic=False)
CHUNK_SIZE = 128 # temporal context 128*512 samples ~ 4sec


