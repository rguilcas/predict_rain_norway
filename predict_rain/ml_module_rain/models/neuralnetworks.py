import torch.nn as nn
import math
import torch
from ml_module_rain.models.individualblocks import ResidualBlock, ConvolutionBlock, MLP

def get_neural_network(config):
    NN = CNN_MLP(feature_height=config['feature_height'], 
             feature_width=config['feature_width'],
             input_channels=config['num_channels'],
             output_neurons=config['num_classes'],
             CNN_number_of_layers=config['num_conv_layer'],
             CNN_output_channels_first_layer=4,
             CNN_base_module = config['CNN_layer_module'],
             MLP_hidden_layers_neuron_number =  config['MLP_hidden_layers_neuron_number'], 
             dropout_MLP =config['dropout_MLP'],
             use_residuals=config['use_skip_connections'],
             activation_function=config['activation_function'],
             global_kwargs_encoder=dict(batch_norm= config['batch_norm_CNN'], dropout=config['dropout_CNN']),
             kwargs_per_layer_encoder = [],)
    return NN




class CNN_MLP(nn.Module):
    def __init__(self, 
                 feature_height, 
                 feature_width,
                 input_channels,
                 output_neurons,
                 CNN_number_of_layers=3,
                 CNN_output_channels_first_layer=16,
                 CNN_channels_increase_per_layer=2,
                 CNN_base_module = 'doubleconv',
                 MLP_hidden_layers_neuron_number = [128,512,512,128],
                 dropout_MLP = 0.1,
                 use_residuals=True,
                 activation_function = 'ReLU',
                 global_kwargs_encoder=dict(),
                 kwargs_per_layer_encoder = [],
                 ):
        super(CNN_MLP, self).__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.activation_function = activation_function
        self.CNN = Encoder(input_channels = input_channels, 
                           num_layers=CNN_number_of_layers, 
                           output_channels_first_layer=CNN_output_channels_first_layer,
                           channels_increase_per_layer=CNN_channels_increase_per_layer,
                           downsample_mode = CNN_base_module,
                           use_residuals=use_residuals,
                           global_kwargs= global_kwargs_encoder,
                           kwargs_per_layer = kwargs_per_layer_encoder, 
                           activation_function=self.activation_function)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, feature_height, feature_width)
            cnn_output = self.CNN(dummy_input)
            linear_size = cnn_output.view(1, -1).shape[1]
        
        self.linear_size = linear_size
        self.MLP = MLP(self.linear_size, 
                       output_neurons,
                       hidden_layers_neuron_number = MLP_hidden_layers_neuron_number,
                       dropout = dropout_MLP,
                       activation_function=self.activation_function
                      )
    def forward(self, x):
        x = self.CNN(x)
        x = x.view(-1, self.linear_size)
        x = self.MLP(x)
        return x



class Encoder(nn.Module):
    def __init__(self, input_channels, num_layers,
                 output_channels_first_layer=16,
                 channels_increase_per_layer=2,
                 use_residuals = True, # can be list
                 downsample_mode="conv+pool", # Can be a list
                 global_kwargs=dict(),
                 kwargs_per_layer = [],
                 activation_function = 'ReLU'):
        super().__init__()
        self.activation_function = activation_function
        self.num_layers = num_layers
        if type(downsample_mode)==str:
            downsample_modes = [downsample_mode for _ in range(self.num_layers)]
        if not isinstance(use_residuals, list):
            use_residuals = [use_residuals for _ in range(self.num_layers)]
        if not global_kwargs:
            if not kwargs_per_layer:
                kwargs_per_layer = [dict() for _ in range(self.num_layers)]
        else:
            kwargs_per_layer = [global_kwargs for _ in range(self.num_layers)]
        for layer in range(self.num_layers):
            in_channels_layer = output_channels_first_layer*(channels_increase_per_layer**(layer-1)) if layer>0 else input_channels
            out_channels_layer = output_channels_first_layer*(channels_increase_per_layer**(layer)) if layer>0 else output_channels_first_layer
            if use_residuals: 
                module =  ResidualBlock(ConvolutionBlock, in_channels_layer, out_channels_layer,downsample_mode = downsample_modes[layer], residual_activation_function=self.activation_function,
                                        activation_function=self.activation_function, **kwargs_per_layer[layer])
            else:
                module =  ConvolutionBlock(in_channels_layer, out_channels_layer,downsample_modes[layer], activation_function=self.activation_function,
                                           **kwargs_per_layer[layer])

            setattr(self, f"convblock{layer}", module)
    def forward(self, x):
        out = x
        for layer in range(self.num_layers):
            out = getattr(self, f"convblock{layer}")(out)
        return out



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