import torch
import librosa
import numpy as np
from config import *
from simple_model import SENetv0
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize

##load data
dataset = torch.load(os.path.join(DATADIR,'processed/dataset-speech.pt'))


#load model to cuda
model = SENetv0().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_fn = torch.nn.MSELoss()

training_set = []
print('preparing training_set')
for record in range(len(dataset[0])):
    noisy_spectrum = dataset[0][record]
    clean_spectrum = dataset[1][record]
    feat_in = np.zeros((noisy_spectrum.shape[0] - CHUNK_SIZE , CHUNK_SIZE, noisy_spectrum.shape[1]))
    feat_out = np.zeros((noisy_spectrum.shape[0] - CHUNK_SIZE, noisy_spectrum.shape[1]))
    for frame in range(noisy_spectrum.shape[0] - CHUNK_SIZE):
        feat_in[frame, :, :] = noisy_spectrum[frame : frame + CHUNK_SIZE, :]
        feat_out[frame, :] = clean_spectrum[CHUNK_SIZE + frame, :]
    training_set.append((feat_in, feat_out))


for epoch in range(100):
    _loss = 0
    for feat_in, feat_out in training_set:
        # The i/o is too big, we may consider normalization here 
        au_in = torch.from_numpy(feat_in).permute(0, 2, 1).cuda().float()
        au_out = torch.from_numpy(feat_out).cuda().float()
        pred_out = torch.squeeze(model(au_in))
        loss = loss_fn(au_out, pred_out)
        _loss += loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(_loss)
