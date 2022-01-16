import torch.nn as nn
import torch

class ChunkData(nn.Module):

    def __init__(self, chunk_size):
        super(ChunkData, self).__init__()

        self.chunk_size = chunk_size
    
    def forward(self, dt):
        with torch.no_grad():

            time, freq = dt['mixed_mag'].shape

            device = dt['mixed_mag'].device

            dt['x'] = torch.zeros((time - self.chunk_size, self.chunk_size, freq), device= device)

            dt['y'] = torch.zeros((time - self.chunk_size, freq), device= device)

            for i in range(time - self.chunk_size):
                
                dt['x'][i, :, :] = dt['mixed_mag'][i : i + self.chunk_size, :]

                # dt['y'][i, :] = dt['noise_mag'][i + self.chunk_size, :]#predict noise
                dt['y'][i, :] = dt['clean_mag'][i + self.chunk_size, :]#predict speech
            
            dt['x'] = dt['x'].permute(0, 2, 1)

        return dt
