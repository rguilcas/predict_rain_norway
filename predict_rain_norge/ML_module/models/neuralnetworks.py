import torch.nn as nn
from collections import OrderedDict


def get_activation(name: str):
    activations = {"ReLU": nn.ReLU(inplace=False),
                   "LeakyReLU": nn.LeakyReLU(inplace=False),
                   "GELU": nn.GELU()}
    return activations.get(name, nn.ReLU(inplace=False))

def get_batch_norm(use_bn: bool, C: int):
    return nn.BatchNorm2d(C) if use_bn else nn.Identity()

# def pad_to_even(x):
#     # right/bottom pad to mimic ceil-style downsampling on odd H/W
#     ph = x.shape[-2] & 1
#     pw = x.shape[-1] & 1
#     return F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x


class ConvolutionLayer(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 act="ReLU", 
                 use_bn=True, 
                 p_drop=0.0):
        super().__init__()
        self.convolution = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=use_bn is False)
        self.activation= get_activation(act)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.batchnorm= get_batch_norm(use_bn, out_ch)
        self.dropout= nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
    def forward(self, x):
        out = self.convolution(x)
        out = self.activation(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.batchnorm(out)
        return out

class CNN(nn.Module):
    def __init__(self, 
                 num_conv_layers, 
                 first_in_ch, 
                 first_out_channels, 
                 channels_increase_per_layer, 
                 max_channels=256,
                 act="ReLU", 
                 use_bn=True, 
                 p_drop=0.0, 
                 drop_out_start_after_layer=0):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        prev_out_ch = first_in_ch
        for layer in range(num_conv_layers):
            out_ch = int(first_out_channels * (channels_increase_per_layer ** layer))
            out_ch = min(out_ch, max_channels)
            module = ConvolutionLayer(
                prev_out_ch, 
                out_ch, 
                act=act, 
                use_bn=use_bn, 
                p_drop=p_drop if layer>drop_out_start_after_layer else 0.0,  # no dropout in first layers
            )
            setattr(self, f"convlayer{layer+1}", module)
            prev_out_ch = out_ch

    def forward(self, x):
        for layer in range(self.num_conv_layers):
            x = getattr(self, f"convlayer{layer+1}")(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, 
                 input_neurons, 
                 output_neurons,
                 hidden_layers_neuron_number = [128,128],
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
                list_modules += [("activation",get_activation(activation_function)),
                                 ("dropout", nn.Dropout(p=dropout))]
            layer = nn.Sequential(OrderedDict(list_modules))
            setattr(self, f"fc{i}", layer )

    def forward(self, x):
        for i in range(1, self.number_layers+1):
            x = getattr(self, f"fc{i}" )(x)
        return x


class CNN_MLP(nn.Module):
    def __init__(self,
                 CNN_num_conv_layers, 
                 CNN_first_in_ch, 
                 CNN_first_out_channels, 
                 CNN_channels_increase_per_layer, 
                 activation_function="ReLU", 
                 CNN_max_channels=256,
                 use_bn=True, 
                 CNN_p_drop=0.0, 
                 CNN_drop_out_start_after_layer=0,
                 MLP_output_neurons=7,
                 MLP_hidden_layers_neuron_number = [128,128],
                 MLP_dropout = 0):
        super(CNN_MLP, self).__init__()
        self.cnn = CNN(num_conv_layers=CNN_num_conv_layers, 
                       first_in_ch=CNN_first_in_ch, 
                       first_out_channels=CNN_first_out_channels, 
                       channels_increase_per_layer=CNN_channels_increase_per_layer, 
                       max_channels=CNN_max_channels,
                       act=activation_function, 
                       use_bn=use_bn, 
                       p_drop=CNN_p_drop, 
                       drop_out_start_after_layer=CNN_drop_out_start_after_layer)
        self.flatten = nn.Flatten()
        self.cnn_out_image_size =(128/(2**self.cnn.num_conv_layers))**2   # Assuming input size is 128x128
        self.cnn_out_channels = getattr(self.cnn, f"convlayer{self.cnn.num_conv_layers}" ).convolution.out_channels
        self.linear_size = int(self.cnn_out_channels*self.cnn_out_image_size)
        self.mlp = MLP(self.linear_size, 
                       MLP_output_neurons,
                       hidden_layers_neuron_number = MLP_hidden_layers_neuron_number,
                       dropout = MLP_dropout,
                       activation_function=activation_function,)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x