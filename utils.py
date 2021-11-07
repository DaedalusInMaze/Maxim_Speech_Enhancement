import os

from tqdm import tqdm

import urllib

import tarfile

import zipfile

import librosa

import numpy as np

def makedir_exist_ok(dirpath):
    """create the folder if it does not exist"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def gen_bar_updater():
    """generate progress bar for downloading file """
    pbar = tqdm(total = None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url, root, filename=None, md5=None):
    """
    download data from url to local

    - url: url of data

    - root: local folder path

    >>> os.path.basename("https://www.openslr.org/resources/12/dev-clean.tar.gz")
    'dev-clean.tar.gz'

    >>> os.path.expanduser('~')              
    'C:\\Users\\username'

    The urllib.request.urlretrieve ()  downloads the remote data directly to the local.

    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    try:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath, reporthook = gen_bar_updater())
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath,
                                       reporthook = gen_bar_updater())
        else:
            raise e


def rmdirfiles(directory):
    """remove files under the directory"""
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            fname = os.path.join(dirpath, filename)
            os.chmod(fname, 0o777)
            os.remove(fname)


def extract_archive(from_path, to_path=None, remove_finished=False):
    """
    extract zip or gz file

    >>> os.path.dirname('SE\processed')
    'SE'
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith('.tar.gz'):
        with tarfile.open(from_path, 'r:gz', errorlevel = 1) as tar:
            for dir in os.listdir(path = to_path):
                if os.path.isdir(os.path.join(to_path, dir)):
                    rmdirfiles(os.path.join(to_path, dir))
            tar.extractall(path = to_path)

    elif from_path.endswith('.zip'):
        with zipfile.ZipFile(from_path, "r") as zip_ref:
            zip_ref.extractall(path = to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

def download_and_extract_archive(url, download_root, extract_root=None, filename=None, md5=None, remove_finished=False):
    """
    dowmload data from url to local, and extract the zip file.

    - url: url of data

    - root: local folder path

    - filename: downloaded filename

    - md5: useless

    - remove_finished: delete zip files
    """
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    archive = os.path.join(download_root, filename)

    if not os.path.isfile(archive):
        download_url(url, download_root, filename, md5)
    else:
        print(f'{filename} exists, no need to download again!')

    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)



def file_count(folder, ext='.flac'):
    """
    count the number of the flac files that are under the folder

    - folder: path
    """
    file_cnt = 0
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in sorted(filenames):
            if filename.endswith(ext):
                file_cnt += 1
    return file_cnt



def resample(folder, sr, ext='.flac', max_prog=100):
    """
    Load audio file as a list of floating point time series.

    Audio will be automatically resampled to the given rate 

    - folder: path

    - sr: sampling_rate for librosa.load

    - max_prog: 1-100%, use percents of all data 
    """
    resampled = []
    file_cnt = 0
    num_files = file_count(folder, ext)
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in sorted(filenames):
            file_cnt += 1
            progress = (100 * file_cnt / num_files)
            if filename.endswith(ext):
                fname = os.path.join(dirpath, filename)
                data, samplerate = librosa.load(fname, sr = sr)
                print(f'\r{int(10 * progress) / 10}%: {fname}, {samplerate} {data.shape}',
                      end = " ")
                resampled.append(data)
                if progress >= max_prog:
                    break
        else:
            continue
        break
    print('\r')
    return resampled


def snr_mixer(clean, noise, snr):
    """
    mix and scale clean speech and noise
    """
    # Normalizing to rms equal to 1
    rmsclean = np.mean(clean[:] ** 2) ** 0.5
    rmsnoise = np.mean(noise[:] ** 2) ** 0.5
    noisyspeech = []

    if rmsclean != 0 and rmsnoise != 0:
        scalarclean = 1 / rmsclean
        clean = clean * scalarclean
        scalarnoise = 1 / rmsnoise
        noise = noise * scalarnoise

        # Set the noise level for a given SNR
        cleanfactor = 10 ** (snr / 20)
        noisyspeech = cleanfactor * clean + noise
        noisyspeech = noisyspeech / (scalarnoise + cleanfactor * scalarclean)
        scaled_clean = cleanfactor * clean / (scalarnoise + cleanfactor * scalarclean)
        scaled_noise = noise / (scalarnoise + cleanfactor * scalarclean)

        valid = True

    else:
        valid = False
    return noisyspeech, valid, scaled_clean, scaled_noise


def quantize_audio(data, num_bits=8):
    """
    Quantize audio
    """

    step_size = 1.0 / 2 ** (num_bits)
    max_val = 2 ** (num_bits) - 1
    q_data = np.round(data / step_size)
    q_data = np.clip(q_data, 0, max_val)

    return np.uint8(q_data)