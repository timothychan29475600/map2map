import torch
import torch.nn as nn

class Conv3D(nn.Module):
    def __init__(self, in_chan, out_chan=None,padding=0,stride=1,kernel_size=(1,1,1)):
        super().__init__()
        
        if out_chan is None:
            out_chan = in_chan # Set out_chan in case it's none
        
        # Stride, kernel size and padding 
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = self.get_padding(padding)
        
        # Make the conv kernel, activation and 
        self.conv = nn.Conv3d(in_chan,out_chan,
                              padding=self.padding,
                              kernel_size=self.kernel_size,
                              stride=self.stride)
        
        self.activate = nn.LeakyReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_chan,track_running_stats=True)
        
    def get_padding(self,padding):
        if type(padding) is int:
            return padding
        elif type(padding) is str:
            if padding == 'same':
                assert sum(k%2 for k in self.kernel_size) == 3, 'Kernel size must be all odd'
                return tuple([k//2 for k in self.kernel_size])
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        self.activate(x)
        return x
        

class NaiveInception(nn.Module):
    def __init__(self,in_chan, out_chan=None,pool=False):
        
        super().__init__()
        
        assert not(out_chan is None), "NaiveInception: out_chan cannot be None" 
        assert type(out_chan) is list and len(out_chan) in [3,4], "NaiveInception: out_chan should be a list with 3/4 elements"
        
        self.in_chan = in_chan
        self.out_chan = out_chan
        
        self.conv_1x1x1 = Conv3D(in_chan,out_chan[0],kernel_size=(1,1,1),padding='same')
        self.conv_3x3x3 = Conv3D(in_chan,out_chan[1],kernel_size=(3,3,3),padding='same')
        self.conv_5x5x5 = Conv3D(in_chan,out_chan[2],kernel_size=(5,5,5),padding='same')
        
        self.avg_pool = nn.AvgPool3d(kernel_size=(1,1,1))
        self.avg_conv = Conv3D(in_chan,out_chan[3],kernel_size=(1,1,1),padding='same')
        
        
    def forward(self,x):
        branch_1x1x1 = self.conv_1x1x1(x)
        branch_3x3x3 = self.conv_3x3x3(x)
        branch_5x5x5 = self.conv_5x5x5(x)
        branch_pool  = self.avg_conv(self.avg_pool(x))
        
        branches = [branch_1x1x1,branch_3x3x3,branch_5x5x5,branch_pool]
        
        print(*[b.shape for b in branches])

        
        return torch.cat(branches,dim=1)
        
