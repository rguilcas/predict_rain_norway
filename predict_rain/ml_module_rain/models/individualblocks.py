import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---- helpers ---------------------------------------------------------------

def get_act(name: str):
    return {"ReLU": nn.ReLU(inplace=False),
            "LeakyReLU": nn.LeakyReLU(inplace=False),
            "GELU": nn.GELU(),
            }.get(name, nn.ReLU(inplace=False))

def get_norm(use_bn: bool, C: int):
    return nn.BatchNorm2d(C) if use_bn else nn.Identity()

def pad_to_even(x):
    # right/bottom pad to mimic ceil-style downsampling on odd H/W
    ph = x.shape[-2] & 1
    pw = x.shape[-1] & 1
    return F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x

# ---- feature blocks --------------------------------------------------------

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, act="ReLU", use_bn=True, p_drop=0.0):
        super().__init__()
        self.convolution = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=use_bn is False)
        self.batchnorm= get_norm(use_bn, out_ch)
        self.activation= get_act(act)
        self.dropout= nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        
    def forward(self, x): 
        out = self.convolution(x)
        out = self.batchnorm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, act="ReLU", use_bn=True, p_drop=0.0):
        super().__init__()
        self.conv1 = SingleConv(in_ch, out_ch, act=act, use_bn=use_bn, p_drop=p_drop)
        self.conv2 = SingleConv(out_ch, out_ch, act=act, use_bn=use_bn, p_drop=p_drop)
    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ResidualBlock(nn.Module):
    """(Single/Double)Conv + residual"""
    def __init__(self, in_ch, out_ch, conv_multiple="double",
                 act="ReLU", use_bn=True, p_drop=0.0, ceil=True, conv_down_kernel=3):
        super().__init__()
        convfeat = DoubleConv if conv_multiple == "double" else SingleConv
        self.mainconvolution = convfeat(in_ch, out_ch, act=act, use_bn=use_bn, p_drop=p_drop)
        
        proj = nn.Conv2d(in_ch, out_ch, 1, bias=use_bn is False)
        self.projection = nn.Sequential(OrderedDict([('projection_block', proj),('batchnorm',get_norm(use_bn, out_ch))]),)
        self.activation = get_act(act)
        
    def forward(self, x):
        out1 = self.mainconvolution(x)
        out2 = self.projection(x)
        return self.activation(out1+out2)

# ---- downsampling op (pure down, no conv unless asked) ---------------------

class Downsample(nn.Module):
    """
    mode: 'maxpool', 'avgpool', 'strideconv', or None
    kernel_stride: kernel/stride for pool, or stride for strideconv
    conv_kernel: kernel for strideconv (2 mimics your old 2x2; 3 is common)
    ceil: if True, mimic ceil_mode semantics on odd H/W via right/bottom pad
    """
    def __init__(self, C, mode="maxpool", kernel_stride=2, conv_kernel=3, ceil=True, use_bn=True, act="ReLU"):
        super().__init__()
        self.mode = mode
        self.ceil = ceil
        if mode is None:
            self.down = nn.Identity()
        elif mode == "maxpool":
            self.down = nn.MaxPool2d(kernel_stride, kernel_stride, ceil_mode=ceil)
        elif mode == "avgpool":
            self.down = nn.AvgPool2d(kernel_stride, kernel_stride, ceil_mode=ceil)
        elif mode == "strideconv":
            # stride-2 conv as a downsampler; keep norm+act for stability
            self.down = nn.Conv2d(C, C, conv_kernel, stride=2, padding=conv_kernel//2, bias=use_bn is False)
        else:
            raise ValueError(f"{mode} should be one of 'strideconv','maxpool' or 'avgpool'")

    def forward(self, x):
        if self.mode in ("strideconv", None):
            return self.down(x)
        if self.ceil:
            x = pad_to_even(x)
        return self.down(x)
    

# ---- unified blocks --------------------------------------------------------

class DownBlock(nn.Module):
    """(Single/Double)Conv → Downsample"""
    def __init__(self, in_ch, out_ch, conv_multiple="double", down_mode="maxpool",
                 act="ReLU", use_bn=True, p_drop=0.0, ceil=True, conv_down_kernel=3):
        super().__init__()
        feat = DoubleConv if conv_multiple == "double" else SingleConv
        self.convolution = feat(in_ch, out_ch, act=act, use_bn=use_bn, p_drop=p_drop)
        self.down = Downsample(out_ch, mode=down_mode, conv_kernel=conv_down_kernel,
                               ceil=ceil, use_bn=use_bn, act=act)
    def forward(self, x):
        x = self.convolution(x)
        x = self.down(x)
        return x

class ResidualDownBlock(nn.Module):
    """(Single/Double)Conv → Downsample"""
    def __init__(self, in_ch, out_ch, conv_multiple="double", down_mode="maxpool",
                 act="ReLU", use_bn=True, p_drop=0.0, ceil=True, conv_down_kernel=3):
        super().__init__()
        self.residual_convolution = ResidualBlock(in_ch, out_ch, conv_multiple=conv_multiple,
                 act=act, use_bn=use_bn, p_drop=p_drop, ceil=ceil, conv_down_kernel=conv_down_kernel)
        self.down = Downsample(out_ch, mode=down_mode, conv_kernel=conv_down_kernel,
                               ceil=ceil, use_bn=use_bn, act=act)
    def forward(self, x):
        out = self.residual_convolution(x)
        out = self.down(out)
        return out
    
    

# class ResidualDownBlockConcat(nn.Module):
#     """Residual ((Single/Double)Conv) + matching downsample on both paths, then concat."""
#     def __init__(self, in_ch, out_ch, conv_multiple="double", down_mode="maxpool",
#                  act="ReLU", use_bn=True, p_drop=0.0, ceil=True, conv_down_kernel=3):
#         super().__init__()
#         assert out_ch >= in_ch, "For concat skip, out_ch must be >= in_ch"
#         add_ch = out_ch - in_ch  # channels added by the main branch

#         feat_cls = DoubleConv if conv_multiple == "double" else SingleConv
#         # Main branch creates the *extra* channels
#         self.feat = feat_cls(in_ch, add_ch, act=act, use_bn=use_bn, p_drop=p_drop)
#         self.down_main = Downsample(add_ch, mode=down_mode, conv_kernel=conv_down_kernel,
#                                     ceil=ceil, use_bn=use_bn, act=act)

#         # Skip branch keeps in_ch channels but downsamples spatially to match main branch
#         if down_mode == "strideconv":
#             proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=2, bias=use_bn is False)
#             skip = nn.Identity()
#         elif down_mode in ("maxpool", "avgpool"):
#             proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=use_bn is False)
#             skip = Downsample(in_ch, mode=down_mode, ceil=ceil)  # only spatial downsample
#         elif down_mode is None:
#             proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=use_bn is False)
#             skip = nn.Identity()
#         else:
#             raise ValueError(f"Unknown down_mode: {down_mode}")

#         self.proj = nn.Sequential(proj, get_norm(use_bn, in_ch))
#         self.skip_down = skip
#         self.act = get_act(act)

#         self.out_ch = out_ch  # for reference

#     def forward(self, x):
#         y = self.down_main(self.feat(x))        # [N, add_ch, H/2, W/2] (or same H/W if no downsample)
#         s = self.skip_down(self.proj(x))        # [N, in_ch,  H/2, W/2]
#         out = torch.cat([y, s], dim=1)          # concat along channels -> [N, add_ch+in_ch, ...] == out_ch
#         # Optional: sanity checks during development
#         # assert out.shape[1] == self.out_ch
#         return self.act(out)


# class ResidualDownBlockConcatNoProj(nn.Module):
#     """Residual ((Single/Double)Conv) + matching downsample on both paths, then concat."""
#     def __init__(self, in_ch, out_ch, conv_multiple="double", down_mode="maxpool",
#                  act="ReLU", use_bn=True, p_drop=0.0, ceil=True, conv_down_kernel=3):
#         super().__init__()
#         assert out_ch >= in_ch, "For concat skip, out_ch must be >= in_ch"
#         add_ch = out_ch - in_ch  # channels added by the main branch

#         feat_cls = DoubleConv if conv_multiple == "double" else SingleConv
#         # Main branch creates the *extra* channels
#         self.feat = feat_cls(in_ch, add_ch, act=act, use_bn=use_bn, p_drop=p_drop)
#         self.down_main = Downsample(add_ch, mode=down_mode, conv_kernel=conv_down_kernel,
#                                     ceil=ceil, use_bn=use_bn, act=act)

#         # Skip branch keeps in_ch channels but downsamples spatially to match main branch
#         if down_mode == "strideconv":
#             proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=2, bias=use_bn is False)
#             skip = nn.Identity()
#         elif down_mode in ("maxpool", "avgpool"):
#             proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=use_bn is False)
#             skip = Downsample(in_ch, mode=down_mode, ceil=ceil)  # only spatial downsample
#         elif down_mode is None:
#             proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=use_bn is False)
#             skip = nn.Identity()
#         else:
#             raise ValueError(f"Unknown down_mode: {down_mode}")

#         self.proj = nn.Sequential(proj, get_norm(use_bn, in_ch))
#         self.skip_down = skip
#         self.act = get_act(act)

#         self.out_ch = out_ch  # for reference

#     def forward(self, x):
#         y = self.down_main(self.feat(x))        # [N, add_ch, H/2, W/2] (or same H/W if no downsample)
#         s = self.skip_down(x)        # [N, in_ch,  H/2, W/2]
#         out = torch.cat([y, s], dim=1)          # concat along channels -> [N, add_ch+in_ch, ...] == out_ch
#         # Optional: sanity checks during development
#         # assert out.shape[1] == self.out_ch
#         return self.act(out)



#### CONVOLUTIONNALBLOCKS




# class ConvolutionBlock(nn.Module):
#     def __init__(self, input_channels, output_channels, 
#                  downsample_mode="conv+pool", 
#                  batch_norm=False, 
#                  dropout=0,
#                  activation_function='ReLU'):
#         super().__init__()
#         self.down_block = get_downsample_block(downsample_mode, input_channels, output_channels)
#         self.batch_norm_true = batch_norm 
#         self.dropout_p = dropout 
#         match activation_function:
#             case 'ReLU':
#                 self.activation_function = nn.ReLU()
#             case 'LeakyReLU':
#                 self.activation_function = nn.LeakyReLU()
#         if self.batch_norm_true : self.batch_norm = nn.BatchNorm2d(output_channels)  
#         if self.dropout_p > 0 : self.dropout = nn.Dropout2d(p=self.dropout_p)

#     def forward(self, x):
#         out = self.downsample_block(x)
#         if self.batch_norm_true : out = self.batch_norm(out)
#         out = self.activation_function(out)
#         if self.dropout_p > 0 : out = self.dropout(out)
#         return out


# def get_downsample_block(mode, input_channels, output_channels, **kwargs):
#     if mode == "conv+pool":
#         return DownsampleConvPool(input_channels, output_channels,**kwargs)
#     elif mode == "convstride":
#         return DownsampleStrideConv(input_channels, output_channels, **kwargs)
#     elif mode == "doubleconv":
#         return DownsampleDoubleConv(input_channels, output_channels, **kwargs)
#     elif mode == "doubleconv+pool":
#         return DownsampleDoubleConvPool(input_channels, output_channels, **kwargs)
    
#     else:
#         raise ValueError(f"Unknown mode: {mode}")
    
# class DownsampleConvPool(nn.Module):
#     """Conv(stride=1) + MaxPool(2x2, stride=2, ceil_mode=True)"""
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
#             nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
#         )

#     def forward(self, x):
#         return self.block(x)
    
# class DownsampleDoubleConvPool(nn.Module):
#     """2xConv(stride=1) + MaxPool(2x2, stride=2, ceil_mode=True)"""
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
#             nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
#         )

#     def forward(self, x):
#         return self.block(x)

# class DownsampleStrideConv(nn.Module):
#     """Conv(stride=2)"""
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.block = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)

#     def forward(self, x):
#         return self.block(x)

# class DownsampleDoubleConv(nn.Module):
#     """Conv(stride=1) + Conv(kernel_size=2, stride=2), with optional padding to match ceil_mode"""
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=2, stride=2, padding=0)

#     def forward(self, x):
#         x = self.conv1(x)
#         # Pad if H or W is odd, to match ceil_mode behavior
#         if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
#             x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
#         x = self.conv2(x)
#         return x



# ##### Residual block

# class ResidualBlock(nn.Module):
#     def __init__(self, base_module, input_channels, output_channels, 
#                  residual_activation_function = 'ReLU',
#                  **base_module_kwargs):
#         super(ResidualBlock, self).__init__()
#         self.main = base_module(input_channels, output_channels, **base_module_kwargs)

#         self.skip_connection = nn.Identity()
#         if input_channels != output_channels:
#             self.skip_connection = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        
#         self.pool_match = False
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         match residual_activation_function:
#             case 'ReLU':
#                 self.activation = nn.ReLU()
#             case 'LeakyReLU':
#                 self.activation = nn.LeakyReLU()

#     def forward(self, x):
#         identity = self.skip_connection(x)
#         if identity.shape[-2] % 2 != 0 or identity.shape[-1] % 2 != 0:
#             identity = F.pad(identity, (0, identity.shape[-1] % 2, 0, identity.shape[-2] % 2))
#         identity = self.pool(identity)
#         out = self.main(x)
#         out = self.activation(out + identity)
#         return out
    
class ContextBlock(nn.Module):
    """
    Dilation-only residual block (no downsampling).
    3x3 (dilation=d) -> Act -> 3x3 (dilation=1) -> add & Act
    """
    def __init__(self, ch, dilation=2, use_bn=True, activation='ReLU', p_dropout=0.0):
        super().__init__()
        Act = getattr(nn, activation)
        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(ch) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(ch) if use_bn else nn.Identity()

        self.act = Act(inplace=False) if 'inplace' in Act.__init__.__code__.co_varnames else Act()
        self.do = nn.Dropout2d(p_dropout) if p_dropout and p_dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.c1(x); h = self.bn1(h); h = self.act(h)
        h = self.c2(h); h = self.bn2(h)
        h = self.do(h)
        return self.act(x + h)

class GlobalAvgPool(nn.Module):
    def forward(self, x):
        # [B, C, H, W] -> [B, C]
        return x.mean(dim=(2, 3))
    
### MLP blocks

class MLP(nn.Module):
    def __init__(self, 
                 input_neurons, 
                 output_neurons,
                 hidden_layers_neuron_number = [128,512,512,128],
                 dropout = 0,
                 activation_function='ReLU',
                 ):
        super(MLP, self).__init__()
        self.number_layers = len(hidden_layers_neuron_number) + 1
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.neurons = [input_neurons] + hidden_layers_neuron_number + [output_neurons]
        
        for i in range(1,len(self.neurons)):
            list_modules = [("linearlayer",nn.Linear(self.neurons[i-1], self.neurons[i]))]
            if i < self.number_layers:
                list_modules += [("activation",get_act(activation_function)),
                                 ("dropout", nn.Dropout(p=dropout))]
            layer = nn.Sequential(OrderedDict(list_modules))
            setattr(self, f"fc{i}", layer )

    def forward(self, x):
        for i in range(1, self.number_layers+1):
            x = getattr(self, f"fc{i}" )(x)
        
        return x
