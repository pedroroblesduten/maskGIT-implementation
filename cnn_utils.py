import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class NonLocal(nn.Module):
    """ 
    NonLocal Block from: 
    Non Local Neural Networks: https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, in_c):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32,
                                       num_channels=in_c,
                                       eps=ie-6, affine=True)

        self.Q = nn.Conv2d(in_channels=in_c,
                           out_channels=in_c,
                           kernel_size = 1,
                           stride = 1,
                           padding = 0)

        self.K = nn.Conv2d(in_channels=in_c,
                           out_channels=in_c,
                           kernel_size = 1,
                           stride = 1,
                           padding = 0)

        self.V = nn.Conv2d(in_channels=in_c,
                           out_channels=in_c,
                           kernel_size = 1,
                           stride = 1,
                           padding = 0)
        
        self.conv_out = nn.Conv2d(in_channels=in_c,
                           out_channels=in_c,
                           kernel_size = 1,
                           stride = 1,
                           padding = 0)


    def forward(self, inp):
        x = self.group_norm(inp)
        b, c, h, w = x.shape
        d_k = sqrt(c)

        Q = self.Q(x).reshape(b, c, h*w)
        Q = Q.permute(0, 2, 1)
        K = self.K(x).reshape(b, c, h*w)
        V = self.V(x).reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn/d_k
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, w, h)
        A = self.conv_out(A)

        output = inp + A

        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_c = in_channels
        self.out_c= out_channels
        self._res_block = nn.Sequential(
            nn.GroupNorm(num_groups=32,
                        num_channels=in_channels,
                        eps=ie-6, affine=True),
            Swish(),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.GroupNorm(num_groups=32,
                        num_channels=out_channels,
                        eps=ie-6, affine=True),
            Swish(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )
        self.channel_up = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

    def forward(self, x):
        res = self._res_block(x)
        if self.in_channels != self.out_channels:
            x = self.channel_up(x)
        out = res + x
        return out



class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)



