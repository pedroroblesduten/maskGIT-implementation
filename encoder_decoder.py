
import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecure from the original JAX implementation from Google Research, but here in PyTorch
# https://github.com/google-research/maskgit/blob/main/maskgit/nets/vqgan_tokenizer.py


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.group_norm1 = nn.GroupNorm(num_groups=32,
                                       num_channels=in_c,
                                       eps=1e-6, affine=True)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_c,
                               out_channels=out_c,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.group_norm2 = nn.GroupNorm(num_groups=32,
                                       num_channels=out_c,
                                       eps=1e-6, affine=True)

        self.conv2 = nn.Conv2d(in_channels=out_c,
                               out_channels=out_c,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.channel_up = nn.Conv2d(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.group_norm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.group_norm2(x)
        x = self.relu(x)
        x = self.conv2(x)        
        if self.in_c != self.out_c:
            res = self.channel_up(res)
            out = res + x
        else:
            out = res + x
        return out

class ResidualStack(nn.Module):
    def __init__(self, in_ch, res_ch, num_residual_layers):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.res_layers = nn.ModuleList([
            ResidualBlock(in_ch, res_ch)
            for _ in range(self.num_residual_layers)
        ])

    def forward(self, x):
        for res_layer in self.res_layers:
            x = res_layer(x)
        return x

class upSample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch,
                              kernel_size=3,
                              stride=1,
                              padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.)
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.verbose = args.verbose
        self.mult_ch = [1, 1, 2, 2]
        self.num_blocks= len(self.mult_ch)
        self.conv_1 = nn.Conv2d(3, 128,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.layers_1 = nn.ModuleList()
        in_channel = 128
        for i in range(self.num_blocks):
            out_channel = 128*self.mult_ch[i]
            for _ in range(args.num_res_blocks):
                self.layers_1.append(ResidualBlock(in_channel, out_channel))
                in_channel = out_channel
            if i < (self.num_blocks - 2):
                self.layers_1.append(nn.Conv2d(out_channel, out_channel, 4, 2, 1))

        self.layers_2 = ResidualStack(in_channel, in_channel, args.num_res_blocks)
        self.group_norm = nn.GroupNorm(num_groups=32,
                                       num_channels=in_channel,
                                       eps=1e-6, affine=True)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channel, args.latent_dim, 1)

    def forward(self, x):
        x = self.conv_1(x)
        if self.verbose:
            print(f'Shape after conv_1; {x.shape}')
        for layer1 in self.layers_1:
            x = layer1(x)
        if self.verbose:
            print(f'Shape after layers1: {x.shape}')
        x = self.layers_2(x)
        if self.verbose:
            print(f'Shape after layers2: {x.shape}')
        x = self.group_norm(x)
        if self.verbose:
            print(f'Shape after groupnorm {x.shape}')
        x = self.conv_2(x)
        if self.verbose:
            print(f'Shape after conv2: {x.shape}')

        return x 


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mult_ch = [1, 1, 2, 2]
        self.num_blocks = len(self.mult_ch)

        self.conv_in = nn.Conv2d(args.latent_dim, 512,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_blocks = ResidualStack(512, 512, args.num_res_blocks)
        self.layers = nn.ModuleList()
        in_channel = 512
        for i in reversed(range(self.num_blocks)):
            out_channel = 128*self.mult_ch[i]
            for _ in range(args.num_res_blocks):
                self.layers.append(ResidualBlock(in_channel, out_channel))
                in_channel = out_channel
            if i > 1:
                self.layers.append(upSample(in_channel))
        self.group_norm = nn.GroupNorm(num_groups=32,
                                       num_channels=in_channel,
                                       eps=1e-6, affine=True)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(in_channel, 3,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        for layer in self.layers:
            x = layer(x)
        x = self.group_norm(x)
        x = self.relu(x)
        x = self.conv_out(x)

        return x         

