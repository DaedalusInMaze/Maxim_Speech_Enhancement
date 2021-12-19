import os
from tqdm import tqdm
import urllib
import tarfile
import zipfile
import librosa
import numpy as np
import soundfile as sf
import torch

sr = 16000
folder = 'testaudio'
DATA = []
savedir = 'SE/raw_speech/test.pt'

for filename in os.listdir(folder):
    fname = os.path.join(folder, filename)
    data, samplerate = librosa.load(fname, sr = sr)
    DATA.append(data)
    
torch.save(DATA, savedir)
