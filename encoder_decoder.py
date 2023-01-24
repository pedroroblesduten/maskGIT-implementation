import torch
import torch.nn as nn
from cnn_utils import Swish, NonLocal, UpSampleBlock, DownSampleBlock

# Architecure from the original JAX implementation from Google Research, but here in PyTorch
# https://github.com/google-research/maskgit/blob/main/maskgit/nets/vqgan_tokenizer.py


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        self.in_c = in_c
        self.out_c = out_c
        self.group_norm = nn.GroupNorm(num_groups=32,
                                       num_channels=in_ch,
                                       eps=1e-6, affine=True)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_c,
                               out_channels=in_c,
                               kernel_size=3
                               stride=1
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=in_c,
                               out_channels=in_c,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.channel_up = nn.Conv2d(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.group_norm(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.group_norm(x)
        x = self.relu(x)
        x = self.conv2(x)        
        if self.in_c !+ self.out_c:
            res = self.channel_up(res)
        out = res + x

class ResidualStack(nn.Module):
    def __init__(self, in_channels, res_channels, num_residual_layers):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.res_layers = nn.ModuleList([
            ResidualBlock(in_channels, res_channels)
            for _ in range(self.num_residual_layers)
        ])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self.res_layers[i](x)
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mult_ch = [1, 1, 2, 2, 4]
        self.conv_1 = nn.Conv2d(3, 128,
                      kernel_size=3,
                      stride=1,
                      padding=1)

        self.layers_1 = nn.ModuleList()
        for i in range(arg.num_blocks):
            in_channels = 128*self.mult_ch[i]
            self.layers.append(ResidualStack(in_channels, in_channels, args.num_res_blocks))
            if i < blocks - 1:
                self.layers.append(nn.Conv2d(in_channels, in_channels, 4, 2, 1))
        self.layers_2 = ResidualStack(in_channels, in_channels, args.num_res_blocks)
        self.group_norm = nn.GroupNorm(num_groups=32,
                                       num_channels=in_ch,
                                       eps=1e-6, affine=True)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels,args.latent_dim, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layers_1(x)
        for layer in self.layers_1:
            x = layer(x)
        for layer in self.layers_2:
            x = layer(x)
        x = self.group_norm(x)
        x = self.conv_2(x)

        return x





            
