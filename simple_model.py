import torch.nn as nn



class SENetv0(nn.Module):

    # output = 161 STFT feats
    def __init__(self, num_channels=161, dimensions=(161, 1), bias=False, **kwargs):
        super().__init__() ## 继承nn.Module的属性

        self.project = nn.Sequential(

            nn.Conv1d(257, 256, 9), # in : 161 x 128; out: 256 x 120
            nn.ReLU(inplace=True),   
            
            nn.MaxPool1d(2),# in : 256 x 120; out: 256 x 60           
            nn.Conv1d(256, 128, 9),# in : 256 x 60; out: 128 x 52
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            #########################################################################
            nn.Conv1d(128, 128, 9, padding=4),# in : 128 x 52; out: 128 x 52
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 9, padding=4),# in : 128 x 52; out: 128 x 52
            nn.ReLU(),
            nn.BatchNorm1d(128),
            #########################################################################
            nn.MaxPool1d(2),# in : 128 x 52; out: 128 x 26
            
            nn.Conv1d(128, 96, 9),# in : 128 x 26, out: 96 x 18
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.MaxPool1d(2), # in : 96 x 18, out: 96 x 9            
            nn.Conv1d(96, 257, 9), # in : 96 x 9, out: 161 x 1
            nn.ReLU()
        )



    def forward(self, x):
        x = self.project(x)
        return x