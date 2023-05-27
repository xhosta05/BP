
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict

def dummy_model(in_planes = 3, out_planes = 64, kernel_size = 5, activation_fn = None, batch_normalization=False):


	model= nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size),
		nn.ReLU(),	
		nn.Linear(out_planes, 45),
	)

	return model


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
        
"""Inspired by https://amaarora.github.io/2020/09/13/unet.html"""
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
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out
"""
	model = nn.Sequential(
		  nn.Conv2d(1,20,5),
		  nn.ReLU(),
		  nn.Conv2d(20,64,5),
		  nn.ReLU()
        )
		nn.ConvTranspose2d(out_planes, in_planes, kernel_size=kernel_size),
		nn.ReLU(),	
        
		nn.BatchNorm2d(out_planes) if batch_normalization else ...,
		activation_fn() if activation_fn is not None else ..."""
		

class Conv_Model_1l(torch.nn.Module):
    def __init__(self, in_channels, out_classes = 3, inner_dia=64):
        super(Conv_Model_1l, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=inner_dia, kernel_size=4, stride=2)
        self.act1 = torch.nn.ReLU()
        self.dense = torch.nn.Linear(in_features=inner_dia*4 , out_features=out_classes)
        
        self.fl = torch.nn.Flatten()
        self.d = nn.Dropout(p=0.3)
        
        # self.layer2 = torch.nn.Conv1d(in_channels=inner_dia, out_channels=out_classes, kernel_size=1)
        
		# nn.Linear(out_planes, 45),
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.fl(x)
        x = self.dense(x)
        
        log_probs = torch.nn.functional.log_softmax(x, dim=1)

        return log_probs


