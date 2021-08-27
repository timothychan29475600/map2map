import torch
import torch.nn as nn

from .conv import ConvBlock, ResBlock
from .narrow import narrow_by


class Resnet(nn.Module):
    def __init__(self, in_chan, out_chan, mid_chan=64, **kwargs):
        """Resnet like network

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
        self.convs = nn.Sequential(
            ConvBlock(in_chan, mid_chan, seq='CA'),
            ResBlock(mid_chan, mid_chan, seq='CBACBA'),
            ResBlock(mid_chan, mid_chan, seq='CBACBA'),
            ResBlock(mid_chan, mid_chan, seq='CBACBA'),
            ResBlock(mid_chan, mid_chan, seq='CBACBA'),
            ConvBlock(mid_chan, out_chan, seq='CAC')
        )

    def forward(self, x):
        x = self.convs(x)
        return x
