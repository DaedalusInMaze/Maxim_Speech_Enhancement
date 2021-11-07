import torch

import librosa

import numpy as np

from config import *

from simple_model import SENetv0

from tqdm import tqdm

##load data
dataset = torch.load('SE/processed/dataset-speech.pt')

print('data shape is: ',dataset[1].shape)

clean_snippet = dataset[1][:1280]

noisy_snippet = dataset[0][:, 127, :]
# print(noisy_snippet.shape)
noisy_angle = dataset[2][:, 127, :]
# print(noisy_angle.shape)
clean_snippet = dataset[1]
# print(clean_snippet.shape)

noisy_recon = clean_snippet * (np.cos(noisy_angle) + 1j * np.sin(noisy_angle))
x = librosa.istft(noisy_recon.T, hop_length=WIN_HOP, win_length=WIN_LEN, window='hanning', center=True, dtype=None, length=None)


#load model to cuda
model = SENetv0().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
optimizer.zero_grad()
loss_fn = torch.nn.MSELoss()
_input, _output, angles, _lab = dataset

#training
for au in tqdm(range(_input.shape[0])):
    au_in = torch.from_numpy(_input[0].T).cuda().unsqueeze(0).float()
    au_out = torch.from_numpy(_output[0]).cuda().float()
    pred_out = model(au_in)
    loss = loss_fn(pred_out, au_out)
    loss.backward()
    optimizer.step()