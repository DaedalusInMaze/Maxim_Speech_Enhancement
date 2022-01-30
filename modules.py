import torch.nn as nn


class CNN2DBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(5, 7), dilation=1, stride=(1, 2), padding=0):

        super(CNN2DBlock, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation),

            nn.BatchNorm2d(channel_out),

            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):

        return self.f(x)

class TCNN2DBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(5, 7), dilation=1, stride=(1, 2), padding=0, dropout=False, output_padding=1):

        super(TCNN2DBlock, self).__init__()

        self.f = nn.Sequential(
            nn.ConvTranspose2d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation,
                output_padding=output_padding),

            nn.BatchNorm2d(channel_out),

            nn.LeakyReLU(negative_slope=0.1)
        )

        self.d = dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.f(x)
        if self.d:
            x = self.dropout(x)
        return x



class MaxPoolConv2dBNReLU(nn.Module):
    def __init__(self, in_channels=1, out_channels= 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine'):
        super(MaxPoolConv2dBNReLU, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.f(x)

class Conv2dBNReLU(nn.Module):
    def __init__(self, in_channels=1, out_channels= 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine'):
        super(Conv2dBNReLU, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.f(x)

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels=1, out_channels= 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine'):
        super(ConvTranspose2d, self).__init__()

        self.f = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.f(x)