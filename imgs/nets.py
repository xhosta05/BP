
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_of_conv_blocks = 1, act_fn = None, pad_to_conserve_dimensions=False) -> None:
        super().__init__()

        padding = 0
        if pad_to_conserve_dimensions:
            padding = int((kernel_size-1) / 2)

        modules = [('conv0', nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding))]

        for i in range(1, num_of_conv_blocks):
            if act_fn is not None:
                modules.append((f'act_fn{i}', act_fn()))
            modules.append( (f'conv{i}', nn.Conv2d(out_channels, out_channels, kernel_size, padding = padding)) )

        self.module = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.module(x)
        

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
#         assert(len(chs) == len(kernel_sizes))

        self.enc_blocks = nn.ModuleList([
            EncoderBlock(chs[i], chs[i+1], 3, num_of_conv_blocks=2, act_fn=nn.ReLU)
            for i in range(len(chs)-1)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
#         assert(len(chs) == len(kernel_sizes))

        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([
            EncoderBlock(chs[i], chs[i+1], 3, num_of_conv_blocks=2, act_fn=nn.ReLU)
            for i in range(len(chs)-1)
        ])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572), head=None):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        if head==None:
        	self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        else:
        	self.head        = head
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

