import torch.nn as nn
import math
import torch

def get_neural_network(config):
    match config['CNN_layer_module']:
        case 'ConvLayerStride1_ConvLayerStride2':
            CNN_module = ConvLayerStride1_ConvLayerStride2
        case 'ConvLayerStride1MaxPool':
            CNN_module = ConvLayerStride1MaxPool
        case 'ConvLayerStride2NoMaxPool':
            CNN_module = ConvLayerStride2NoMaxPool
        case 'ConvLayerStride1_ConvLayerStride2_dropout':
            CNN_module = ConvLayerStride1_ConvLayerStride2_dropout
    NN = CNN_MLP(feature_height=config['feature_height'], 
                feature_width=config['feature_width'],
                input_channels=config['num_channels'],
                output_neurons=config['num_classes'],
                CNN_number_of_layers=config['num_conv_layer'],
                CNN_base_module = CNN_module,
                MLP_hidden_layers_neuron_number = config['MLP_hidden_layers_neuron_number'], 
                use_residual = config['use_skip_connections']
                )
    return NN
    
class ConvLayerStride1MaxPool(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 activation_function=nn.ReLU(),
                 size_conv_kernel=3,
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
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2,padding=1)

    def forward(self, x):
        output = self.conv_layer(x)
        output = self.bn2(output)
        output = self.maxpool(output)
        output = self.activation_function(output)
        return output



                 
                 
class ConvLayerStride1_ConvLayerStride2_dropout(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 activation_function=nn.ReLU(),
                 size_conv_kernel=3,
                 dropout_p=.3,
                 dropout_min_channels=128,
                 ):
        super(ConvLayerStride1_ConvLayerStride2, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation_function = activation_function
        self.dropout_p = dropout_p
        self.dropout_min_channels=dropout_min_channels
        self.conv_layer1 = nn.Conv2d(self.input_channels, 
                                    self.output_channels, 
                                    kernel_size=size_conv_kernel, 
                                    stride=1, padding=(size_conv_kernel-1)//2)
        self.conv_layer2 = nn.Conv2d(self.output_channels, 
                                    self.output_channels, 
                                    kernel_size=2, 
                                    stride=2, padding=0)
        
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.dropout = nn.Dropout2d(p=self.dropout_p)

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.bn2(output)
        output = self.activation_function(output)
        if self.output_channels>=self.dropout_min_channels:
            output = self.dropout(output)
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
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, feature_height, feature_width)
            cnn_output = self.CNN(dummy_input)
            linear_size = cnn_output.view(1, -1).shape[1]
        
        self.linear_size = linear_size
        self.MLP = MLP(self.linear_size, 
                       output_neurons,
                       hidden_layers_neuron_number = MLP_hidden_layers_neuron_number,
                      )
    def forward(self, x):
        x = self.CNN(x)
        x = x.view(-1, self.linear_size)
        x = self.MLP(x)
        return x




def register_shape_hooks(model, input_shape):
    """Register forward hooks on all modules and print output shapes."""
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            print(f"{name:<50} | Output shape: {tuple(output.shape)}")
        return fn

    for name, module in model.named_modules():
        # Skip the entire model itself
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Run a dummy input through the model
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape)
        model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()