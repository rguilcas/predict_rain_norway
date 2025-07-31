import torch
import torch.nn as nn
import torch.nn.functional as F



#### CONVOLUTIONNALBLOCKS




class ConvolutionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, 
                 downsample_mode="conv+pool", 
                 batch_norm=False, 
                 dropout=0):
        super().__init__()
        self.downsample_block = get_downsample_block(downsample_mode, input_channels, output_channels)
        self.batch_norm_true = batch_norm 
        self.dropout_p = dropout 
        self.activation_function = nn.ReLU()
        if self.batch_norm_true : self.batch_norm = nn.BatchNorm2d(output_channels)  
        if self.dropout_p > 0 : self.dropout = nn.Dropout2d(p=self.dropout_p)

    def forward(self, x):
        out = self.downsample_block(x)
        if self.batch_norm_true : out = self.batch_norm(out)
        out = self.activation_function(out)
        if self.dropout_p > 0 : out = self.dropout(out)
        return out


def get_downsample_block(mode, input_channels, output_channels, **kwargs):
    if mode == "conv+pool":
        return DownsampleConvPool(input_channels, output_channels,**kwargs)
    elif mode == "convstride":
        return DownsampleStrideConv(input_channels, output_channels, **kwargs)
    elif mode == "doubleconv":
        return DownsampleDoubleConv(input_channels, output_channels, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
class DownsampleConvPool(nn.Module):
    """Conv(stride=1) + MaxPool(2x2, stride=2, ceil_mode=True)"""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

    def forward(self, x):
        return self.block(x)

class DownsampleStrideConv(nn.Module):
    """Conv(stride=2)"""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.block(x)

class DownsampleDoubleConv(nn.Module):
    """Conv(stride=1) + Conv(kernel_size=2, stride=2), with optional padding to match ceil_mode"""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        # Pad if H or W is odd, to match ceil_mode behavior
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        x = self.conv2(x)
        return x



##### Residual block

class ResidualBlock(nn.Module):
    def __init__(self, base_module, input_channels, output_channels, **base_module_kwargs):
        super(ResidualBlock, self).__init__()
        self.main = base_module(input_channels, output_channels, **base_module_kwargs)

        self.skip_connection = nn.Identity()
        if input_channels != output_channels:
            self.skip_connection = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        
        self.pool_match = False
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = self.skip_connection(x)
        if identity.shape[-2] % 2 != 0 or identity.shape[-1] % 2 != 0:
            identity = F.pad(identity, (0, identity.shape[-1] % 2, 0, identity.shape[-2] % 2))
        identity = self.pool(identity)
        out = self.main(x)
        out = self.activation(out + identity)
        return out
    

### MLP blocks

class MLP(nn.Module):
    def __init__(self, 
                 input_neurons, 
                 output_neurons,
                 hidden_layers_neuron_number = [128,512,512,128],
                 dropout = 0
                 ):
        super(MLP, self).__init__()
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.neurons = [input_neurons] + hidden_layers_neuron_number + [output_neurons]
        self.layers = nn.ModuleList()
        for i in range(1,len(self.neurons)):
            layer = nn.Linear(self.neurons[i-1], self.neurons[i])
            self.layers.append(layer)
            if i < len(self.neurons)-1 and dropout>0:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    