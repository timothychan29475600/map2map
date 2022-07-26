import torch
import torch.nn as nn

from .conv import ConvBlockDropout, ResBlockDropout
from .narrow import narrow_by


class DustNet(nn.Module):
    def __init__(self, in_chan, out_chan, bypass=None, **kwargs):
        """V-Net like network

        Note:

        Global bypass connection adding the input to the output (similar to
        COLA for displacement input and output) from Alvaro Sanchez Gonzalez.
        Enabled by default when in_chan equals out_chan

        Global bypass, under additive symmetry, effectively obviates --aug-add

        Non-identity skip connection in residual blocks
        """
        super().__init__()

        # activate non-identity skip connection in residual block
        # by explicitly setting out_chan
        self.p = float(kwargs['mc_dropout_p']) # Change dropout rate to test
        print(f'Dropout probability: {self.p}')
        self.conv_l0 = ResBlockDropout(in_chan, 64, seq='COACOBA',p=self.p)
        self.down_l0 = ConvBlockDropout(64, seq='DOBA',p=self.p)
        self.conv_l1 = ResBlockDropout(64, 64, seq='COBACOBA',p=self.p)
        self.down_l1 = ConvBlockDropout(64, seq='DOBA',p=self.p)

        self.conv_c = ResBlockDropout(64, 64, seq='COBACOBA',p=self.p)

        self.up_r1 = ConvBlockDropout(64, seq='UOBA',p=self.p)
        self.conv_r1 = ResBlockDropout(128, 64, seq='COBACOBA',p=self.p)
        self.up_r0 = ConvBlockDropout(64, seq='UOBA',p=self.p)
        self.conv_r0 = ResBlockDropout(128, out_chan, seq='COAC',p=self.p)

        if bypass is None:
            self.bypass = in_chan == out_chan
        else:
            self.bypass = bypass

    def forward(self, x):
        if self.bypass:
            x0 = x

        y0 = self.conv_l0(x)
        x = self.down_l0(y0)

        y1 = self.conv_l1(x)
        x = self.down_l1(y1)

        x = self.conv_c(x)

        x = self.up_r1(x)
        y1 = narrow_by(y1, 4)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x)

        x = self.up_r0(x)
        y0 = narrow_by(y0, 16)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r0(x)

        if self.bypass:
            x0 = narrow_by(x0, 20)
            x += x0

        return x
