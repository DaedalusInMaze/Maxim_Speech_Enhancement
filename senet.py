import torch
import torch.nn as nn

from modules import *


class SENetv0(nn.Module):

    # output = 161 STFT feats
    def __init__(self, num_channels=257, dimensions=(257, 1), bias=True, **kwargs):
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

    def forward(self, dt):
        x = dt['x']  # in: (batch, frequency, time), out:(batch, frequency, time)
        x = self.project(x) # in:(batch_size, 257, 128) out:(batch_size, 257, 1)
        dt['pred_y'] = torch.squeeze(x)
        return dt



class SENetv1(nn.Module):
    """
    1 torch.Size([100, 257, 128])  ->  torch.Size([100, 768, 128]) # 1x1 conv point-wise?
    2 torch.Size([100, 768, 128])  ->  torch.Size([100, 768, 64])
    3 torch.Size([100, 768, 64])  ->  torch.Size([100, 768, 32])
    4 torch.Size([100, 768, 32])  ->  torch.Size([100, 768, 16])
    5 torch.Size([100, 768, 16])  ->  torch.Size([100, 768, 8])
    6 torch.Size([100, 768, 8])  ->  torch.Size([100, 768, 4])
    7 torch.Size([100, 768, 4])  ->  torch.Size([100, 768, 2])
    8 torch.Size([100, 768, 2])  ->  torch.Size([100, 768, 1])
    9 torch.Size([100, 768, 1])  ->  torch.Size([100, 257, 1])
    """
    def __init__(self, freq_bin = 257, hidden_dim = 768, num_layer = 7, kernel_size = 3):
        super(SENetv1, self).__init__()

        input_layer = CNNBlock(freq_bin, hidden_dim, kernel_size=1)

        down_block = CNNBlock(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)

        pooling_block = nn.MaxPool1d(kernel_size, stride=2, padding=kernel_size//2)

        output_layer = CNNBlock(hidden_dim, freq_bin, kernel_size, padding=kernel_size//2)

        self.encoder = nn.ModuleList()
        self.encoder.append(input_layer)
        for i in range(num_layer):
            self.encoder.append(nn.Sequential(
                down_block,
                pooling_block
            )
        )
        self.encoder.append(output_layer)    

    def forward(self, dt):
        x = dt['x']
        for layer in self.encoder:
            x = layer(x)
        dt['pred_mask'] = torch.squeeze(x)
        return dt


    

    
    
    


class SENetv2(nn.Module):
    """
    1 torch.Size([100, 257, 128])  ->  torch.Size([100, 768, 128])
    2 torch.Size([100, 768, 128])  ->  torch.Size([100, 768, 64])
    3 torch.Size([100, 768, 64])  ->  torch.Size([100, 768, 32])
    4 torch.Size([100, 768, 32])  ->  torch.Size([100, 768, 16])
    5 torch.Size([100, 768, 16])  ->  torch.Size([100, 768, 8])
    6 torch.Size([100, 768, 8])  ->  torch.Size([100, 768, 4])
    7 torch.Size([100, 768, 4])  ->  torch.Size([100, 768, 2])
    8 torch.Size([100, 768, 2])  ->  torch.Size([100, 768, 1])
    9 torch.Size([100, 768, 1])  ->  torch.Size([100, 257, 1])
    """
    def __init__(self, freq_bin = 257, hidden_dim = 768, num_layer = 7, kernel_size = 3):
        super(SENetv2, self).__init__()

        input_layer = CNNBlock(freq_bin, hidden_dim, kernel_size, padding=kernel_size//2)

        down_block = CNNBlock(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)

        pooling_block = nn.MaxPool1d(kernel_size, stride=2, padding=kernel_size//2)

        output_layer = CNNBlock(hidden_dim, freq_bin, kernel_size, padding=kernel_size//2)

        self.encoder = nn.ModuleList()
        self.encoder.append(input_layer)
        for i in range(num_layer):
            self.encoder.append(nn.Sequential(
                down_block,
                pooling_block
            )
        )
        self.encoder.append(output_layer)    

    def forward(self, dt):
        x = dt['x']
        for layer in self.encoder:
            x = layer(x)
        dt['pred_mask'] = torch.squeeze(x)
        return dt

    

class SENetv3(nn.Module):
    """
    chunk_size=16
    """
    def __init__(self, freq_bin = 257, hidden_dim = 768, num_layer = 7, kernel_size = 3):
        super(SENetv3, self).__init__()

        e1 = CNN2DBlock(1, 64, kernel_size= (5, 7), stride= (2, 1), padding= (1, 3))
        e2 = CNN2DBlock(64, 128, kernel_size= (5, 7), stride= (2, 1), padding= (2, 3))
        e3 = CNN2DBlock(128, 256, kernel_size= (5, 7), stride= (2, 1), padding= (2, 3))
        e4 = CNN2DBlock(256, 512, kernel_size= (5, 5), stride= (2, 1), padding= (2, 2))
        e5 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e6 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e7 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e8 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        
        self.encoders = nn.ModuleList([e1, e2, e3, e4, e5, e6, e7, e8])

        d8 = TCNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2), dropout= True)
        d7 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2), dropout= True)
        d6 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2), dropout= True)
        d5 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2), dropout= True)
        d4 = TCNN2DBlock(1024, 256, kernel_size= (5, 5), stride= (2, 1), padding = (2, 2), dropout= True, output_padding=(1,0))
        d3 = TCNN2DBlock(512, 128, kernel_size= (5, 7), stride= (2, 1), padding = (2, 3), dropout= True, output_padding=(1,0))
        d2 = TCNN2DBlock(256, 64, kernel_size= (5, 7), stride= (2, 1), padding = (2, 3), dropout= True, output_padding=(1,0))
        d1 = TCNN2DBlock(128, 1, kernel_size= (5, 7), stride= (2, 1), padding = (1, 3), dropout= True, output_padding=(0,0))  
        self.decoders = nn.ModuleList([d8, d7, d6, d5, d4, d3, d2, d1])

    def forward(self, dt):
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        skip_outputs = []
        for layer in self.encoders:
            x = layer(x)
            skip_outputs.append(x)
        
        skip_output = skip_outputs.pop() 
        first = True
        for layer in self.decoders:
            if first:
                first = False
            else:
                skip_output = skip_outputs.pop()
                x = torch.cat([x, skip_output], dim = 1)
            x = layer(x)
        dt['pred_mask'] = torch.squeeze(x).permute(0, 2, 1)
        return dt


class SENetv4(nn.Module):
    def __init__(self, channel):
        super(SENetv4, self).__init__()

        self.encoder = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=3, stride=1,
            padding=1, dilation=1),

            nn.ReLU(True),

            CNNBlockv2(channel * 1, channel * 2),

            CNNBlockv2(channel * 2, channel * 4),

            CNNBlockv2(channel * 4, channel * 2),

            CNNBlockv2(channel * 2, channel * 1)]
        )
        
        self.decoder = nn.ModuleList(
            [CNNBlockv2(channel * 1, channel * 2),

            CNNBlockv2(channel * 2, channel * 4),

            CNNBlockv2(channel * 4, channel * 2),

            CNNBlockv2(channel * 2, channel * 1),

            nn.Dropout2d(0.2, True),

            CNNBlockv2(channel * 1, 1)]
        )

    def forward(self, dt):
        
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])

        skip = []

        for layer in self.encoder:

            x = layer(x)
            skip.append(x)

        for layer in self.decoder:

            s = skip.pop()
            x = layer(x + s)
        
        x = x[:,:,:257,:]
        dt['pred_mask'] = torch.squeeze(x,1).permute(0, 2, 1)

        return dt
class SENetv5(nn.Module):
    def __init__(self, channel):
        super(SENetv5, self).__init__()

        self.encoder = nn.ModuleList(
            [CNNBlockv2(1, channel * 1),

            CNNBlockv2(channel * 1, channel * 2),

            CNNBlockv2(channel * 2, channel * 4),

            CNNBlockv2(channel * 4, channel * 8), 
            
            CNNBlockv2(channel * 8, channel * 16), 
            ]
        )
        
        self.decoder = nn.ModuleList(
            [CNNBlockv2(channel * 16, channel * 8),

            CNNBlockv2(channel * 8, channel * 4),

            CNNBlockv2(channel * 4, channel * 2),

            CNNBlockv2(channel * 2, channel * 1),

            CNNBlockv2(channel * 1, 1)]
        )

    def forward(self, dt):
        
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])

        skip = []

        for layer in self.encoder:

            x = layer(x)
            skip.append(x)

        for layer in self.decoder:

            s = skip.pop()
            x = layer(x + s)
        
        dt['pred_mask'] = torch.squeeze(x).permute(0, 2, 1)
    
        return dt

class SENetv6(nn.Module):
    def __init__(self, channel):
        super(SENetv6, self).__init__()

        self.encoder = nn.ModuleList(
            [CNNBlockv2(1, channel, kernel_size=5),

            CNNBlockv2(channel, channel, kernel_size=5),
            ]
        )
        
        self.decoder = nn.ModuleList(
            [CNNBlockv2(channel, channel, kernel_size=5),

            CNNBlockv2(channel, 1, kernel_size=5)]
        )

    def forward(self, dt):
        
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        
        skip = []
        for layer in self.encoder:

            x = layer(x)
            skip.append(x)

        for layer in self.decoder:
            
            s = skip.pop()
            x = layer(x + s)

        dt['pred_mask'] = torch.squeeze(x).permute(0, 2, 1)
        return dt

class SENetv7(nn.Module):
    """
    chunk_size=16
    """
    def __init__(self, ):
        super(SENetv7, self).__init__()

        e1 = CNN2DBlockv2(1, 64, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e2 = CNN2DBlockv2(64, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e3 = CNN2DBlockv2(128, 256, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e4 = CNN2DBlockv2(256, 256, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e5 = CNN2DBlockv2(256, 256, kernel_size= 3, stride= 1, padding= 1, maxpool=False)

        
        self.encoders = nn.ModuleList([e1, e2, e3, e4, e5])

        d5 = CNN2DBlockv2(256, 256, kernel_size= 3, stride= 1, padding= 1, maxpool=False)
        d4 = TCNN2DBlockv2(512, 256, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d3 = TCNN2DBlockv2(512, 128, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d2 = TCNN2DBlockv2(256, 64, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d1 = TCNN2DBlockv2(128, 1, kernel_size= 3, stride= 2, padding = 1, output_padding=1)  
        self.decoders = nn.ModuleList([d5, d4, d3, d2, d1])

    def forward(self, dt):
        
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        e1 = self.encoders[0](x)

        e2 = self.encoders[1](e1)

        e3 = self.encoders[2](e2)

        e4 = self.encoders[3](e3)

        e5 = self.encoders[4](e4)

        d5 = self.decoders[0](e5)

        d4 = self.decoders[1](torch.cat([d5, e4], dim=1))

        d3 = self.decoders[2](torch.cat([d4, e3], dim=1))

        d2 = self.decoders[3](torch.cat([d3, e2], dim=1))

        d1 = self.decoders[4](torch.cat([d2, e1], dim=1))
        dt['pred_mask'] = torch.squeeze(d1).permute(0, 2, 1)
        return dt

    
class SENetv8(nn.Module):
    """
    chunk_size=16
    """
    def __init__(self, ):
        super(SENetv8, self).__init__()

        e1 = CNN2DBlockv2(1, 64, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e2 = CNN2DBlockv2(64, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e3 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e4 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e5 = CNN2DBlockv2(128, 256, kernel_size= 3, stride= 1, padding= 1, maxpool=False)

        
        self.encoders = nn.ModuleList([e1, e2, e3, e4, e5])

        d5 = CNN2DBlockv2(256, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=False)
        d4 = TCNN2DBlockv2(256, 128, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d3 = TCNN2DBlockv2(256, 128, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d2 = TCNN2DBlockv2(256, 64, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d1 = TCNN2DBlockv2(128, 1, kernel_size= 3, stride= 2, padding = 1, output_padding=1)  
        self.decoders = nn.ModuleList([d5, d4, d3, d2, d1])

    def forward(self, dt):
        
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        e1 = self.encoders[0](x)

        e2 = self.encoders[1](e1)

        e3 = self.encoders[2](e2)

        e4 = self.encoders[3](e3)

        e5 = self.encoders[4](e4)

        d5 = self.decoders[0](e5)

        d4 = self.decoders[1](torch.cat([d5, e4], dim=1))

        d3 = self.decoders[2](torch.cat([d4, e3], dim=1))

        d2 = self.decoders[3](torch.cat([d3, e2], dim=1))

        d1 = self.decoders[4](torch.cat([d2, e1], dim=1))
        dt['pred_mask'] = torch.squeeze(d1).permute(0, 2, 1)
        return dt
    

class SENetv10(nn.Module):
    def __init__(self, ):
        super(SENetv10, self).__init__()

        e1 = CNN2DBlockv2(1, 64, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e2 = CNN2DBlockv2(64, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e3 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e4 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e5 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=False)

        
        self.encoders = nn.ModuleList([e1, e2, e3, e4, e5])

        d5 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=False)
        d4 = TCNN2DBlockv2(256, 128, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d3 = TCNN2DBlockv2(256, 128, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d2 = TCNN2DBlockv2(256, 64, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d1 = TCNN2DBlockv2(128, 1, kernel_size= 3, stride= 2, padding = 1, output_padding=1)  
        self.decoders = nn.ModuleList([d5, d4, d3, d2, d1])

    def forward(self, dt):
        
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        e1 = self.encoders[0](x)

        e2 = self.encoders[1](e1)

        e3 = self.encoders[2](e2)

        e4 = self.encoders[3](e3)

        e5 = self.encoders[4](e4)

        d5 = self.decoders[0](e5)

        d4 = self.decoders[1](torch.cat([d5, e4], dim=1))

        d3 = self.decoders[2](torch.cat([d4, e3], dim=1))

        d2 = self.decoders[3](torch.cat([d3, e2], dim=1))

        d1 = self.decoders[4](torch.cat([d2, e1], dim=1))

        dt['pred_mask'] = torch.squeeze(d1,1).permute(0, 2, 1)
        return dt

class SENetv11(nn.Module):
    """
    chunk_size=16 version 3 without dropout
    """
    def __init__(self, freq_bin = 257, hidden_dim = 768, num_layer = 7, kernel_size = 3):
        super(SENetv11, self).__init__()

        e1 = CNN2DBlock(1, 64, kernel_size= (5, 7), stride= (2, 1), padding= (1, 3))
        e2 = CNN2DBlock(64, 128, kernel_size= (5, 7), stride= (2, 1), padding= (2, 3))
        e3 = CNN2DBlock(128, 256, kernel_size= (5, 7), stride= (2, 1), padding= (2, 3))
        e4 = CNN2DBlock(256, 512, kernel_size= (5, 5), stride= (2, 1), padding= (2, 2))
        e5 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e6 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e7 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e8 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        
        self.encoders = nn.ModuleList([e1, e2, e3, e4, e5, e6, e7, e8])

        d8 = TCNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d7 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d6 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d5 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d4 = TCNN2DBlock(1024, 256, kernel_size= (5, 5), stride= (2, 1), padding = (2, 2), output_padding=(1,0))
        d3 = TCNN2DBlock(512, 128, kernel_size= (5, 7), stride= (2, 1), padding = (2, 3), output_padding=(1,0))
        d2 = TCNN2DBlock(256, 64, kernel_size= (5, 7), stride= (2, 1), padding = (2, 3), output_padding=(1,0))
        d1 = TCNN2DBlock(128, 1, kernel_size= (5, 7), stride= (2, 1), padding = (1, 3),  output_padding=(0,0))  
        self.decoders = nn.ModuleList([d8, d7, d6, d5, d4, d3, d2, d1])

    def forward(self, dt):
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        skip_outputs = []
        for layer in self.encoders:
            x = layer(x)
            skip_outputs.append(x)
        
        skip_output = skip_outputs.pop() 
        first = True
        for layer in self.decoders:
            if first:
                first = False
            else:
                skip_output = skip_outputs.pop()
                x = torch.cat([x, skip_output], dim = 1)
            x = layer(x)
        dt['pred_mask'] = torch.squeeze(x).permute(0, 2, 1)
        return dt
