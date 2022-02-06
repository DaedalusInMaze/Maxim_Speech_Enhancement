import os, glob

import librosa

import numpy as np

import soundfile as sf

import torch

from tqdm import tqdm

import pysepm

def load_noise(noise_paths):
    print('loading noise')
    return {i : torch.load(path) for i, path in tqdm(enumerate(noise_paths))}

def save_wav(path, wav, fs):
    """
    Save .wav file.
    Argument/s:
        path - absolute path to save .wav file.
        wav - waveform to be saved.
        fs - sampling frequency.
    """
    wav = np.squeeze(wav)
    sf.write(path, wav, fs)

def get_audio_path_list(dir, ext):
    """
    >>> train_path = os.path.join(root, 'train')
    >>> train_list = get_audio_path_list(train_path, 'flac')
    return全部结尾是ext的文件地址list
    """
    wav_path_list = []
    wav_path_list.extend(glob.glob(os.path.join(dir, f"**/*.{ext}"), recursive=True))
    return wav_path_list


def load_audio(path, sr= 16000):
    """
    读取audio
    return numpy array
    """

    waveform, sr = librosa.load(path, sr=sr)

    return waveform


def truncate_pad(sequence, max_len):
    """
    根据max_len填充或截取序列
    """
    length = len(sequence)
    
    if length < max_len:

        pad = (0, max_len - length) #pad 0 个在array前面， pad max_len - length个到array后面

        sequence = np.pad(sequence, pad)
    
    else:

        sequence = sequence[:max_len]

    return sequence


def snr_mixer(clean, noise, snr):
    """
    mix and scale clean speech and noise
    """
    # Normalizing to rms equal to 1
    rmsclean = np.mean(clean[:] ** 2) ** 0.5
    rmsnoise = np.mean(noise[:] ** 2) ** 0.5

    cleanfactor = 10 ** (snr / 20)
    
    if rmsnoise != 0:
        valid = True
        a = rmsclean/(rmsnoise*cleanfactor)
        # Set the noise level for a given SNR
        noisyspeech = clean + a*noise
        return valid, noisyspeech, a*noise
    else:
        valid = False
        return valid, 0, 0
    

def quantize_spectrum(data, num_bits=8):
    """
    Quantize spectrum
    """

    step_size = 1.0 / 2 ** (num_bits)
    max_val = 2 ** (num_bits) - 1
#     q_data = np.round(data / step_size)
#     q_data = np.clip(q_data, 0, max_val)

#     return np.uint8(q_data)
    q_data = torch.round(data / step_size)
    q_data = torch.clamp(q_data, 0, max_val)
    return q_data.to(torch.uint8)

def normalize_quantized_spectrum(data, num_bits=8):
    """
    Normalize Quantized spectrum
    """
    return data/ (2 ** (num_bits) - 1)


def generate_test_files(path, noise_path, snr):
    
    noise = torch.load(noise_path)
    
    clean = load_audio(path = path)
    
    while True:
    
        noise_start = np.random.randint(0, noise.shape[0] - clean.shape[0] + 1)

        noise_snippet = noise[noise_start : noise_start + clean.shape[0]]

        valid, mixed, noise = snr_mixer(clean, noise_snippet, snr)
        
        if valid:
            break
        
    return {'mixed': mixed, 'clean': clean, 'noise': noise}


def evaluation(clean_speech, pred_speech, sr = 16_000):
    """
    return pesq, stoi, segSNR
    """
    return round(pysepm.pesq(clean_speech, pred_speech, 16_000)[1], 2), \
           round(pysepm.stoi.stoi(clean_speech, pred_speech, 16_000), 2), \
           round(pysepm.SNRseg(clean_speech, pred_speech, 16_000), 2)