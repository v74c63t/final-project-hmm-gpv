import torch
import torch.nn as nn

## TODO

# 6 encoders, 5 decoders

class U2Net(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(U2Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO add more args later
    
    def forward(self, x):
        ## TODO
        return