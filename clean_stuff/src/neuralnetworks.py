import torch.nn as nn

class ConvLayerStride1MaxPool(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 activation_function=nn.ReLU(),
                 size_conv_kernel=3
                 ):
        super(ConvLayerStride1MaxPool, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation_function = activation_function
        self.conv_layer = nn.Conv2d(self.input_channels, 
                                    self.output_channels, 
                                    kernel_size=size_conv_kernel, 
                                    stride=1, padding=(size_conv_kernel-1)//2)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        output = self.conv_layer(x)
        output = self.bn2(output)
        output = self.maxpool(output)
        output = self.activation_function(output)
        return output

class ConvLayerStride1_ConvLayerStride2(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 activation_function=nn.ReLU(),
                 size_conv_kernel=3
                 ):
        super(ConvLayerStride1_ConvLayerStride2, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation_function = activation_function
        self.conv_layer1 = nn.Conv2d(self.input_channels, 
                                    self.output_channels, 
                                    kernel_size=size_conv_kernel, 
                                    stride=1, padding=(size_conv_kernel-1)//2)
        self.conv_layer2 = nn.Conv2d(self.output_channels, 
                                    self.output_channels, 
                                    kernel_size=2, 
                                    stride=2, padding=0)
        
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.bn2(output)
        output = self.activation_function(output)
        return output
    
class ConvLayerStride2NoMaxPool(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 activation_function=nn.ReLU(),
                 size_conv_kernel=3
                 ):
        super(ConvLayerStride2NoMaxPool, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation_function = activation_function
        self.conv_layer = nn.Conv2d(self.input_channels, 
                                    self.output_channels, 
                                    kernel_size=size_conv_kernel, 
                                    stride=2, padding=(size_conv_kernel-1)//2)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        output = self.conv_layer(x)
        output = self.bn2(output)
        output = self.activation_function(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, base_module, input_channels, output_channels):
        super(ResidualBlock, self).__init__()
        self.main = base_module(input_channels, output_channels)

        self.skip_connection = nn.Identity()
        if input_channels != output_channels:
            self.skip_connection = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        
        self.pool_match = False
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = self.skip_connection(x)
        identity = self.pool(identity)
        out = self.main(x)
        out = self.activation(out + identity)
        return out
    

class CNN(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels_first_layer=16,
                 channels_increase_per_layer=2,
                 base_module = ConvLayerStride1MaxPool,
                 num_layers=3,
                 activation_function=nn.ReLU(),
                 size_conv_kernel=3,
                 use_residual=False,
                 ):
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.output_layers = [output_channels_first_layer*channels_increase_per_layer**layer for layer in range(num_layers)]
        self.convs = nn.ModuleList()
        self.activation_function = activation_function
        self.size_conv_kernel = size_conv_kernel
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else self.output_layers[i - 1]
            out_ch = self.output_layers[i]
            module = base_module(
                in_ch,
                out_ch,
                activation_function=self.activation_function,
                size_conv_kernel=self.size_conv_kernel            
                )
            
            if self.use_residual:
                module = ResidualBlock(base_module, in_ch, out_ch)
            self.convs.append(module)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class MLP(nn.Module):
    def __init__(self, 
                 input_neurons, 
                 output_neurons,
                 hidden_layers_neuron_number = [128,512,512,128],
                 ):
        super(MLP, self).__init__()
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.neurons = [input_neurons] + hidden_layers_neuron_number + [output_neurons]
        self.layers = nn.ModuleList()
        for i in range(1,len(self.neurons)):
            layer = nn.Linear(self.neurons[i-1], self.neurons[i])
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class CNN_MLP(nn.Module):
    def __init__(self, 
                 feature_height, 
                 feature_width,
                 input_channels,
                 output_neurons,
                 CNN_number_of_layers=3,
                 CNN_output_channels_first_layer=16,
                 CNN_channels_increase_per_layer=2,
                 CNN_base_module = ConvLayerStride1MaxPool,
                 CNN_activation_function=nn.ReLU(),
                 CNN_size_conv_kernel=3,
                 MLP_hidden_layers_neuron_number = [128,512,512,128],
                 use_residual=True,
                 ):
        super(CNN_MLP, self).__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.CNN = CNN(input_channels = input_channels, 
                       num_layers=CNN_number_of_layers, 
                       output_channels_first_layer=CNN_output_channels_first_layer,
                       channels_increase_per_layer=CNN_channels_increase_per_layer,
                       base_module = CNN_base_module,
                       activation_function=CNN_activation_function,
                       size_conv_kernel=CNN_size_conv_kernel,
                       use_residual=use_residual)
        reduced_image_size = self.feature_height//(2**self.CNN.num_layers) * self.feature_width//(2**self.CNN.num_layers)
        last_out_layers = self.CNN.output_layers[-1]
        self.linear_size = reduced_image_size*last_out_layers
        self.MLP = MLP(self.linear_size, 
                       output_neurons,
                       hidden_layers_neuron_number = MLP_hidden_layers_neuron_number,
                      )
    def forward(self, x):
        x = self.CNN(x)
        x = x.view(-1, self.linear_size)
        x = self.MLP(x)
        return x
