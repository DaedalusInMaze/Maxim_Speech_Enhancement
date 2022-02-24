import os

import torch
from tqdm import tqdm

from stft import ISTFT, torch_istft
from utils import evaluation, save_wav


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
        best_snr = 0
        best_pesq = 0
        best_stoi = 0

        for epoch in range(epoch, epoch + epochs + 1):

            train_loss = self._train(epoch)

            loss, pesq, stoi, segsnr = self._valid(epoch)

            
            if save_model:
                if not os.path.exists(kwargs['model_path']):
                    os.mkdir(kwargs['model_path'])
                
                if best_snr < segsnr or best_pesq < pesq or best_stoi < stoi:
                    best_snr = segsnr
                    best_stoi=stoi
                    best_pesq = pesq
                    state_dict = {
                        'epoch': epoch,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }

                    # torch.save(state_dict, os.path.join(kwargs['model_path'], f'{segsnr}_{epoch}_epoch.pth.tar'))
                    torch.save(state_dict, os.path.join(kwargs['model_path'], f'best_model.pth.tar'))
                
            self._test(self.test_loader, epoch, *args, **kwargs)


    
    def _train(self, epoch):

        total_loss = 0
        
        total_pesq, total_stoi, total_segsnr = 0, 0, 0

        train_bar = tqdm(self.train_set)     

        for batch in train_bar:

            train_bar.set_description(f'Epoch {epoch} train')

            for key, value in batch.items():

                batch[key] = value.to(self.device)
        
            self.optimizer.zero_grad()

            batch = self.model(batch)

            loss = self.criterion(batch['pred_mask'], batch['y'])

            total_loss += loss.item()
            
            loss.backward()
            
            self.optimizer.step()
            
#             if epoch > 1:
                
#                 pesq, stoi, segsnr = evaluation(batch['true_y'].numpy(), batch['pred_y'].numpy())
                
#             else:
                
            pesq, stoi, segsnr = 0, 0, 0
                
            total_pesq += pesq
            
            total_stoi += stoi
            
            total_segsnr += segsnr

            train_bar.set_postfix(loss=round(loss.item(),2), pesq=pesq, stoi=stoi, segSNR=segsnr)
        
        lens = len(self.train_set)
        
        average_loss = round(total_loss / lens, 4)
        
        average_pesq = round(total_pesq / lens, 2)
        
        average_stoi = round(total_stoi / lens, 2)
        
        average_segsnr = round(total_segsnr / lens, 2)
               

        print('\tLoss: ', average_loss, 'pesq: ', average_pesq, 'stoi: ', average_stoi, 'sngSNR: ', average_segsnr)
        
        return average_loss


    def _valid(self, epoch):

        total_loss = 0
        
        total_pesq, total_stoi, total_segsnr = 0, 0, 0

        valid_bar = tqdm(self.valid_set)

        for batch in valid_bar:

            valid_bar.set_description(f'Epoch {epoch} valid')

            for key, value in batch.items():

                batch[key] = value.to(self.device).float()
        
            with torch.no_grad():

                self.model.eval()

                batch = self.model(batch, train=False)

                loss = self.criterion(batch['pred_mask'], batch['y'])

                total_loss += loss.item()
                
                pesq, stoi, segsnr = evaluation(batch['true_y'].numpy(), batch['pred_y'].numpy())
            
                total_pesq += pesq

                total_stoi += stoi

                total_segsnr += segsnr

                valid_bar.set_postfix(loss=round(loss.item(),2), pesq=pesq, stoi=stoi, segSNR=segsnr)
        lens = len(self.valid_set)
        
        average_loss = round(total_loss / lens, 4)

        average_loss = round(total_loss / lens, 4)
        
        average_pesq = round(total_pesq / lens, 2)
        
        average_stoi = round(total_stoi / lens, 2)
        
        average_segsnr = round(total_segsnr / lens, 2)
               

        print('\tLoss: ', average_loss, 'pesq: ', average_pesq, 'stoi: ', average_stoi, 'sngSNR: ', average_segsnr)

        return average_loss, average_pesq, average_stoi, average_segsnr

    

    def _test(self, test_loader, epoch, *args, **kwargs):
        
        # test_bar = tqdm(test_loader)
        
        i = 1
        
        for batch in test_loader:
            
            # test_bar.set_description(f'Epoch {epoch} test')
            
            for key, value in batch.items():

                    batch[key] = value.to(self.device).float()

            with torch.no_grad():

                    self.model.eval()

                    batch = self.model(batch, train=False)

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
                
        
        