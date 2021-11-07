SAMPLING_RATE = 16000
URL_SPEECH = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
URL_NOISE = "https://zenodo.org/record/1227121/files/OOFFICE_48k.zip?download=1"

WIN_LEN = int((20*SAMPLING_RATE)/1000) # 20 msec
WIN_HOP = int((10*SAMPLING_RATE)/1000) # 10 msec


frame_length = int((20*SAMPLING_RATE)/1000) # 20 msec
chunk_size = 128 # temporal context 128*10msec ~ 1sec
fft_num = 320
snr = 6
flag = True
stft_len = fft_num//2 + 1