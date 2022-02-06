import torch.nn as nn

class CNN2DBlockv2(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, dilation=1, stride=1, padding=0, maxpool=None):

        super(CNN2DBlockv2, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation),

            nn.BatchNorm2d(channel_out),

            nn.ReLU(),

            
        )
        self.maxpool = maxpool
        self.m = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.f(x)
        if self.maxpool:
            x = self.m(x)
        return x
class TCNN2DBlockv2(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(5, 7), dilation=1, stride=(1, 2), padding=0, output_padding=1):

        super(TCNN2DBlockv2, self).__init__()

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

            nn.ReLU()
        )

        
    def forward(self, x):
        x = self.f(x)
        return x

class CNNBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batchnorm=True, dropout=False):
        super(CNNBlockv2, self).__init__()
        if kernel_size ==5:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                stride=1, padding=padding, dilation=1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                stride=1, padding=padding, dilation=1),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=1, padding=padding, dilation=1
            )
            
        self.bn = None
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        self.d = None
        if dropout:
            self.d = nn.Dropout2d(0.5, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        if self.bn:
            x = self.bn(x)
        if self.d:
            x = self.d(x)
        return x


class TCNNBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCNNBlockv2, self).__init__()
        
        conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
            padding=1, output_padding=1)

        relu = nn.ReLU(True)

        bn = nn.BatchNorm2d(out_channels)

        self.f = nn.Sequential(
            conv1,
            bn,
            relu
        )
    def forward(self, x):
        return self.f(x)


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

class AvgPoolConv2dBNReLU(nn.Module):
    def __init__(self, in_channels=1, out_channels= 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine'):
        super(AvgPoolConv2dBNReLU, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2)
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

class CNNBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, dilation=1, stride=1, padding=0):

        super(CNNBlock, self).__init__()

        self.f = nn.Sequential(
            nn.Conv1d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation),

            nn.BatchNorm1d(channel_out),

            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):

        return self.f(x)