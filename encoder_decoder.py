import torch
import torch.nn as nn
from cnn_utils import ResidualBlock, Swish, NonLocal, UpSampleBlock, DownSampleBlock

class Encoder(nn.Module):
    def __init__(self, latent_dim, i, n_blocks):
        super().__init__()

        self.n_blocks = n_blocks
        self.i = i 
        self.latent_dim = latent_dim

        self.conv_1 = nn.Conv2d(in_channels, 128,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        self.res_block1 = ResidualBlock(128, 128)
        self.down1 = DownSampleBlock(128)
        self.res_block2 = ResidualBlock(128, 128)
        self.down2 = DownSampleBlock(128)
        self.res_block3 = ResidualBlock(128, 256)
        self.down3 = DownSampleBlock(256)
        self.res_block4 = ResidualBlock(256, 256)
        self.down3 = DownSampleBlock(256)
        self.res_block5 = ResidualBlock(256, 512)
        
