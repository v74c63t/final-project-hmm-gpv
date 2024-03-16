"""
This code is adapted from the U-Net paper. See details in the paper:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. 
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad

class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Module that implements 
            - a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        
        Input:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            mid_channels (int): number of channels to use in the intermediate layer    
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels if mid_channels != None else out_channels//2
        
        # print(f"In Channels (DoubleConv): {self.in_channels}")
        # print(f"Out Channels (DoubleConv): {self.out_channels}")
        # print(f"Mid Channels (DoubleConv): {self.mid_channels}")
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, padding=1) #padding = 1 if needed
        self.batchNorm1 = nn.BatchNorm2d(self.mid_channels)
        self.reLu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=3, padding=1) #padding = 1 if needed
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        # raise NotImplementedError


    def forward(self, x):
        """Forward pass through the layers of the helper block"""
        # print(f"Before DoubleConvHelper Shape: {x.shape}")
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.reLu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        # x = self.reLu(x) # Try running w/o this reLu for now
        # print(f"After DoubleConvHelper Shape: {x.shape}")
        return x
        raise NotImplementedError


class Encoder(nn.Module):
    """ Downscale using the maxpool then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxPool = nn.MaxPool2d(kernel_size=2, padding=0) # TODO I don't know what to put into here as the argument for maxpool2d
        self.doubleConv = DoubleConvHelper(in_channels, out_channels)
        # raise NotImplementedError
        # print(f"In Channels (Encoder): {self.in_channels}")
        # print(f"Out Channels (Encoder): {self.out_channels}")

    def forward(self, x):
        # idk this prob wron
        # x = max pool stuff
        # doubleConv = DoubleConvHelper(self.in_channels, self.out_channels)
        # return doubleConv.forward(x)
        # print(f"Before Encoder Shape: {x.shape}")
        x = self.maxPool(x)
        x = self.doubleConv(x)
        # print(f"After Encoder Shape: {x.shape}")
        return x
        raise NotImplementedError


class Decoder(nn.Module):
    """ Upscale using ConvTranspose2d then call double conv helper. """
    def __init__(self, in_channels, out_channels, embedding_size = None): #Added an extra parameter for the sake of our shape error with the last decoder (will force in_channels of 64 to out_channels of 4, but we only need those out channels at the DoubleConvHelper, not at the ConvTranspose
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_size = embedding_size
        # self.convTranspose = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=3) if embedding_size == None else nn.ConvTranspose2d(self.in_channels, self.embedding_size, kernel_size=3) #Specific to last decoder, where the embedding size is used as out_channels for the convTranspose, but the real out_channels is used for the DoubleConvHelper
        self.convTranspose = nn.ConvTranspose2d(self.in_channels, self.in_channels//2, kernel_size=2, stride=2)

        self.doubleConv = DoubleConvHelper(in_channels, out_channels) # Purposely doing double out_channels since I'm (Michael) assuming we're convTransposing then doubleConv-ing
        # raise NotImplementedError
        # print(f"In Channels (Decoder): {self.in_channels}")
        # print(f"Out Channels (Decoder): {self.out_channels}")
    
    def forward(self, x1, x2):
        """ 
        1) x1 is passed through either the upsample or the convtranspose2d
        2) The difference between x1 and x2 is calculated to account for differences in padding
        3) x1 is padded (or not padded) accordingly
        4) x2 represents the skip connection
        5) Concatenate x1 and x2 together with torch.cat
        6) Pass the concatenated tensor through a doubleconvhelper
        7) Return output
        """
        # step 1: replace x1 with the upsampled version of x1
        # print(f"Before convTranspose shape: {x1.shape}")
        x1 = self.convTranspose(x1)
        # print(f"After convTranspose shape: {x1.shape}")
        # input is Channel Height Width, step 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # step 3
        x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        # step 4 & 5: Concatenate x1 and x2
        # print(f"Before X1 shape: {x1.shape}")
        # print(f"Before X2 shape: {x2.shape}")
        cat = torch.cat([x1, x2], dim=1)
        # print(f"Concatenated shape: {cat.shape}")

        # step 6: Pass the concatenated tensor through a doubleconvhelper
        # doubleConvHelper = DoubleConvHelper(self.in_channels, self.out_channels)
        # print(f"Before doubleConv shape: {cat.shape}")
        DoubleConvHelper = self.doubleConv(cat)
        # doubleConv = doubleConvHelper.forward(cat)
        # print(f"AfterdoubleConv shape: {DoubleConvHelper.shape}")

        # step 7: return output
        return DoubleConvHelper
        raise NotImplementedError
    
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders. 
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.

        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is 
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.

        Input:
            in_channels: number of input channels of the image
            of shape (batch, in_channels, width, height)
            out_channels: number of output channels of prediction,
            prediction is shape (batch, out_channels, width//scale_factor, height//scale_factor)
            n_encoders: number of encoders to use in the network (implementing this parameter is
            optional, but it is a good idea to have it as a parameter in case you want to experiment,
            if you do not implement it, you can assume n_encoders=2)
            embedding_size: number of channels to use in the first layer
            scale_factor: number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_encoders = n_encoders
        self.embedding_size = embedding_size
        self.scale_factor = scale_factor

        # print(f"U Net in channels: {self.in_channels}")
        # print(f"U Net out channels: {self.out_channels}")
        # print(f"U Net embedding sizes: {self.embedding_size}")
        
        # print("Inc 2x Conv:")
        self.incDoubleConv = DoubleConvHelper(self.in_channels, self.embedding_size)

        # print("Encoders: ")
        encoders = []
        self.changeable_embedding_size = self.embedding_size #self.changeable_embedding_size is used to be changed so we can keep self.embedding_size the same
        for e in range(self.n_encoders): # matches TA
            # print(f"Encoder #{e}")
            encoders.append(Encoder(self.changeable_embedding_size, 2 * self.changeable_embedding_size))
            self.changeable_embedding_size = 2 * self.changeable_embedding_size

        # print(f"Current changeable_embedding_size: {self.changeable_embedding_size}")
        # print("Decoders:")
        decoders = []
        for d in range(self.n_encoders):
            # self.changeable_embedding_size = self.changeable_embedding_size // 2
            # print(f"Decoder #{d}")
            if d == self.n_encoders - 1:
                # print("Last Decoder")
                decoders.append(Decoder(self.changeable_embedding_size, self.out_channels, self.changeable_embedding_size//2))
            else:
                decoders.append(Decoder(self.changeable_embedding_size, self.changeable_embedding_size//2))
            self.changeable_embedding_size = self.changeable_embedding_size // 2
            

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        # print("Max pooler:")
        self.maxPooler = nn.MaxPool2d(kernel_size=scale_factor)
        # raise NotImplementedError


    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        residuals = []
        
        # print("Forward inc:")
        x = self.incDoubleConv(x)
        
        # print("Forward Encoders:")
        eCount = 1
        for e in self.encoders:
            residuals.append(x)
            # print(f"encoder # {eCount}")
            eCount+=1
            x = e(x)
        
        # x should currently be the last encoded part of the U Net process 
        # print(f"encoders: {len(self.encoders)}")
        # print(f"decoders: {len(self.decoders)}")
        # print(f"residuals: {len(residuals)}")
        
        # print("Forward Decoders: ")
        for index, (d, res) in enumerate(zip(self.decoders, reversed(residuals[:]))):
            # print(f"decoder # {index}")
            # print(f"Residual Shape: {res.shape}")
            x = d(x, res)
        
        # print("Forward Max pooler:")
        return self.maxPooler(x)
        raise NotImplementedError

