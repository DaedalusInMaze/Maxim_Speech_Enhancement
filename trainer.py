from tqdm import tqdm

import torch

import os

from stft import ISTFT, torch_istft

from utils import save_wav

class Trainer():

    def __init__(self, model, train_loader, valid_loader, test_loader, optimizer, criterion, device):

        self.model = model
        self.train_set = train_loader
        self.valid_set = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.count = 0
        
        

    def train(self, epoch, epochs, save_model, *args, **kwargs):
        """
        - epoch: starting epoch
        - epochs: int 玄学
        - save_model: boolean, 保存模型吗？ 保存到models文件夹下
        """
        best_loss = 1_000_000_000

        for epoch in range(epoch, epoch + epochs + 1):

            train_loss = self._train(epoch)

#             loss = self._valid(epoch)

#             if loss < best_loss:

#                 if save_model:

#                     if not os.path.exists(kwargs['model_path']):

#                         os.mkdir(kwargs['model_path'])
            if train_loss < best_loss:
                
                best_loss = train_loss
            
            if save_model:
                if not os.path.exists(kwargs['model_path']):
                    os.mkdir(kwargs['model_path'])

                state_dict = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_loss': best_loss
                }
            
                torch.save(state_dict, os.path.join(kwargs['model_path'], f'{epoch}_epoch.pth.tar'))
                
            self._test(self.test_loader, epoch, *args, **kwargs)


    
    def _train(self, epoch):

        total_loss = 0

        train_bar = tqdm(self.train_set)     

        for batch in train_bar:

            train_bar.set_description(f'Epoch {epoch} train')

            for key, value in batch.items():

                batch[key] = value.to(self.device)
        
            self.optimizer.zero_grad()

            batch = self.model(batch)

            loss = self.criterion(batch['pred_y'], batch['y'])

            total_loss += loss.item()
            
            loss.backward()
            
            self.optimizer.step()

            train_bar.set_postfix(loss=loss.item())
        
        average_loss = round(total_loss / len(self.train_set), 4)

        print('\tLoss:', average_loss)
        
        return average_loss


    def _valid(self, epoch):

        total_loss = 0

        valid_bar = tqdm(self.valid_set)

        for batch in valid_bar:

            valid_bar.set_description(f'Epoch {epoch} valid')

            for key, value in batch.items():

                batch[key] = value.to(self.device).float()
        
            with torch.no_grad():

                self.model.eval()

                batch = self.model(batch)

                loss = self.criterion(batch['pred_y'], batch['y'])

                total_loss += loss.item()

                valid_bar.set_postfix(loss=loss.item())
        
        average_loss = round(total_loss / len(self.valid_set), 4)

        print('\tLoss:', average_loss)

        return average_loss

    

    def _test(self, test_loader, epoch, *args, **kwargs):
        
        # test_bar = tqdm(test_loader)
        
        i = 1
        
        for batch in test_loader:
            
            # test_bar.set_description(f'Epoch {epoch} test')
            
            for key, value in batch.items():

                    batch[key] = value.to(self.device).float()

            with torch.no_grad():

                    self.model.eval()

                    batch = self.model(batch)

                    # iStft = ISTFT(hop_len=kwargs['hop_len'],
                    #               win_len= kwargs['win_len'],
                    #               window= 'hanning',
                    #               device=self.device,
                    #               chunk_size= kwargs['chunk_size'])
                    iStft = torch_istft(n_fft =kwargs['n_fft'],
                                  hop_length=kwargs['hop_len'],
                                  win_length= kwargs['win_len'],
                                  device=self.device,
                                  chunk_size= kwargs['chunk_size'],
                                  transform_type =kwargs['transform_type'],
                                  cnn=kwargs['cnn'],
                                  target= kwargs['target'])

                    batch = iStft(batch)

                    if not os.path.exists(kwargs['recovered_path']):

                        os.mkdir(kwargs['recovered_path'])

                    if self.count < 3:

                        save_wav(path= os.path.join(kwargs['recovered_path'], f'clean_speech_{i}.wav'),
                                 wav= batch['true_y'],
                                 fs= kwargs['fs'])

                        save_wav(path= os.path.join(kwargs['recovered_path'], f'mixed_speech_{i}.wav'),
                                 wav= batch['mixed_y'],
                                 fs= kwargs['fs'])


                        self.count += 1

                    save_wav(path= os.path.join(kwargs['recovered_path'],
                                                f'epoch_{epoch}_recovered_speech_{i}.wav'),
                             wav= batch['pred_y'],
                             fs= kwargs['fs'])
            i += 1
        print(f'\t test files are generated for Epoch {epoch}!')
                
        
        