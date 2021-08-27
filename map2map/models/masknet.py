import torch
import torch.nn as nn
from .incept import Conv3D,NaiveInception

class MaskNet(nn.Module):
    def __init__(self, in_chan, out_chan, **kwargs):
        
        super().__init__()
        
        next_chan = [64,128,32,32]
        self.l1_incept = NaiveInception(in_chan, next_chan)
        
        next_chan = sum(next_chan)
        self.l2_conv = Conv3D(next_chan,next_chan//2,kernel_size=(3,3,3),padding='same')
        
        self.l3_conv = Conv3D(next_chan//2,next_chan//4,kernel_size=(3,3,3),padding='same')
        
        self.l4_conv = Conv3D(next_chan//4,1, kernel_size=(3,3,3),padding='same')

        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        y = self.l1_incept(x)
        y = self.l2_conv(y)
        y = self.l3_conv(y)
        y = self.l4_conv(y)
        return self.sigmoid(y) # Use sigmoid at the end
        
