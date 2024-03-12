import torch
import torch.nn as nn

## TODO

class U2Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # add args, etc