import os

from torchaudio.datasets import LIBRISPEECH

from config import DATADIR


###############################    DOWNLOAD  FILES   #####################################
class SEDataset(LIBRISPEECH):
    def __init__(self, root, types):
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root, types)
        if not os.path.exists(root):
            os.mkdir(root)
        if types == 'train':
            url = 'dev-clean'
        elif types == 'valid':
            url = 'test-clean'
        elif types == 'train_full':
            url = 'train-clean-100'
        super().__init__(root, download=True, url=url)

# SEDataset(DATADIR, 'train')
# SEDataset(DATADIR, 'valid')
SEDataset(DATADIR, 'train_full')
print('downloaded!')
