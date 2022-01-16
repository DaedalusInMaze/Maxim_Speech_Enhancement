import torch
import torch.nn as nn


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

class SENetv1(nn.Module):
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
        dt['pred_y'] = torch.squeeze(x)
        return dt

