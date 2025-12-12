import torch.nn as nn
import math
import torch
from ml_module_rain.models.individualblocks import  MLP, DownBlock, ResidualDownBlock, ContextBlock, GlobalAvgPool

def get_neural_network(config):
    if config['split_MLP_head_per_horizon']:
        if config.get('MLP_multihead_v2', False):
            NN_class = CNN_MLP_multiheads_v2
        else:
            NN_class = CNN_MLP_multiheads
    else:
        if config['use_gap_in_CNN']:
            NN_class = CNN_GAP_MLP
        else:
            NN_class = CNN_MLP
    NN = NN_class(feature_height=config['feature_height'], 
             feature_width=config['feature_width'],
             input_channels=config['num_channels'],
             output_neurons=config['num_classes'],
             CNN_number_of_layers=config['num_conv_layer'],
             CNN_output_channels_first_layer=config['conv1_kernel_number'],
             CNN_downsample_mode = config['CNN_downsample_mode'],
             CNN_conv_multiple = config['CNN_conv_multiple'],
             MLP_hidden_layers_neuron_number =  config['MLP_hidden_layers_neuron_number'], 
             dropout_MLP =config['dropout_MLP'],
             use_residuals=config['use_skip_connections'],
             activation_function=config['activation_function'],
             )
    return NN


class CNN_GAP_MLP(nn.Module):
    def __init__(self, 
                 feature_height, 
                 feature_width,
                 input_channels,
                 output_neurons,
                 CNN_number_of_layers=3,
                 CNN_output_channels_first_layer=16,
                 CNN_channels_increase_per_layer=2,
                 CNN_downsample_mode = 'strideconv',
                 CNN_conv_multiple = 'double',
                 MLP_hidden_layers_neuron_number = [128,512,512,128],
                 dropout_MLP = 0.1,
                 use_residuals=True,
                 activation_function = 'ReLU',
                 CNN_use_bn = True,
                 CNN_p_dropout =0.2
                 ):
        super(CNN_GAP_MLP, self).__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.activation_function = activation_function
        self.CNN = Encoder(input_channels = input_channels, 
                           num_layers=CNN_number_of_layers, 
                           output_channels_first_layer=CNN_output_channels_first_layer,
                           channels_increase_per_layer=CNN_channels_increase_per_layer,
                           downsample_mode=CNN_downsample_mode,    # str or list[str]
                           conv_multiple=CNN_conv_multiple,   
                           use_residuals=use_residuals,
                           use_bn=CNN_use_bn,
                           p_dropout=CNN_p_dropout,
                           activation_function=self.activation_function)
        self.gap = GlobalAvgPool() 
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, feature_height, feature_width)
            cnn_output = self.CNN(dummy_input)
            gap_output = self.gap(cnn_output)
            linear_size = gap_output.shape[1]
        
        self.linear_size = linear_size
        self.MLP = MLP(self.linear_size, 
                       output_neurons,
                       hidden_layers_neuron_number = MLP_hidden_layers_neuron_number,
                       dropout = dropout_MLP,
                       activation_function=self.activation_function
                      )
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.CNN(x)
        x = self.gap(x)
        x = self.MLP(x)
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
                 CNN_downsample_mode = 'strideconv',
                 CNN_conv_multiple = 'double',
                 MLP_hidden_layers_neuron_number = [128,512,512,128],
                 dropout_MLP = 0.1,
                 use_residuals=True,
                 activation_function = 'ReLU',
                 CNN_use_bn = True,
                 CNN_p_dropout =0.2
                 ):
        super(CNN_MLP, self).__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.activation_function = activation_function
        self.CNN = Encoder(input_channels = input_channels, 
                           num_layers=CNN_number_of_layers, 
                           output_channels_first_layer=CNN_output_channels_first_layer,
                           channels_increase_per_layer=CNN_channels_increase_per_layer,
                           downsample_mode=CNN_downsample_mode,    # str or list[str]
                           conv_multiple=CNN_conv_multiple,   
                           use_residuals=use_residuals,
                           use_bn=CNN_use_bn,
                           p_dropout=CNN_p_dropout,
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
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.CNN(x)
        x = self.flatten(x)#.view(-1, self.linear_size)
        x = self.MLP(x)
        return x

class CNN_MLP_multiheads(nn.Module):
    def __init__(self, 
                 feature_height, 
                 feature_width,
                 input_channels,
                 output_neurons,  # should now be number of horizons, e.g., 7
                 CNN_number_of_layers=3,
                 CNN_output_channels_first_layer=16,
                 CNN_channels_increase_per_layer=2,
                 CNN_downsample_mode = 'strideconv',
                 CNN_conv_multiple = 'double',
                 MLP_hidden_layers_neuron_number=[128,512,512,128],
                 dropout_MLP=0.1,
                 use_residuals=True,
                 activation_function='ReLU',
                 CNN_use_bn = True,
                 CNN_p_dropout =0.2,
                ):
        super(CNN_MLP_multiheads, self).__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.activation_function = activation_function
        self.num_horizons = output_neurons  # <-- treat "classes" as "horizons"

        # Shared CNN backbone
        self.CNN = Encoder(
            input_channels=input_channels, 
            num_layers=CNN_number_of_layers, 
            output_channels_first_layer=CNN_output_channels_first_layer,
            channels_increase_per_layer=CNN_channels_increase_per_layer,
            downsample_mode=CNN_downsample_mode,    # str or list[str]
            conv_multiple=CNN_conv_multiple,
            use_residuals=use_residuals,
            use_bn=CNN_use_bn,
            p_dropout=CNN_p_dropout,
            activation_function=self.activation_function
        )

        # Figure out backbone output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, feature_height, feature_width)
            cnn_output = self.CNN(dummy_input)
            linear_size = cnn_output.view(1, -1).shape[1]
        self.linear_size = linear_size

        # One MLP head per horizon
        self.heads = nn.ModuleList([
            MLP(
                input_neurons=self.linear_size,
                output_neurons=1,  # 1 logit per horizon
                hidden_layers_neuron_number=MLP_hidden_layers_neuron_number,
                dropout=dropout_MLP,
                activation_function=self.activation_function
            )
            for _ in range(self.num_horizons)
        ])

    def forward(self, x):
        feats = self.CNN(x)
        feats = feats.view(-1, self.linear_size)

        # Collect logits from each horizon head
        logits = [head(feats) for head in self.heads]  # list of (B, 1)
        logits = torch.cat(logits, dim=1)  # shape: (B, num_horizons)

        return logits
    
class CNN_MLP_multiheads_v2(nn.Module):
    def __init__(self, 
                 feature_height, 
                 feature_width,
                 input_channels,
                 output_neurons,               # number of horizons
                 CNN_number_of_layers=5,       # ↓ recommend 5 instead of 7
                 CNN_output_channels_first_layer=16,
                 CNN_channels_increase_per_layer=2,
                 CNN_downsample_mode='strideconv',
                 CNN_conv_multiple='double',
                 MLP_hidden_layers_neuron_number=[128,512,512,128],
                 dropout_MLP=0.1,
                 use_residuals=True,
                 activation_function='ReLU',
                 CNN_use_bn=True,
                 CNN_p_dropout=0.2,
                 # --- new: context add-on + GAP ---
                 context_dilations=(2,),       # e.g., (2,) or (2,4)
                 context_p_dropout=0.0,
                ):
        super().__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.activation_function = activation_function
        self.num_horizons = output_neurons

        # Shared CNN backbone (your existing Encoder)
        self.CNN = Encoder(
            input_channels=input_channels, 
            num_layers=CNN_number_of_layers, 
            output_channels_first_layer=CNN_output_channels_first_layer,
            channels_increase_per_layer=CNN_channels_increase_per_layer,
            downsample_mode=CNN_downsample_mode,    # str or list[str]
            conv_multiple=CNN_conv_multiple,
            use_residuals=use_residuals,
            use_bn=CNN_use_bn,
            p_dropout=CNN_p_dropout,
            activation_function=self.activation_function
        )

        # Figure out CNN output shape/channels
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, feature_height, feature_width)
            cnn_out = self.CNN(dummy)               # [1, C, H', W']
            out_channels = cnn_out.shape[1]

        # Context add-on: one or more dilated residual blocks (no downsampling)
        self.context = nn.Sequential(*[
            ContextBlock(out_channels, d, use_bn=CNN_use_bn,
                         activation=activation_function, p_dropout=context_p_dropout)
            for d in context_dilations
        ]) if len(context_dilations) > 0 else nn.Identity()

        # Global Average Pooling (replaces flatten)
        self.gap = GlobalAvgPool()
        self.linear_size = out_channels            # after GAP, features are [B, C]

        # One MLP head per horizon (shared encoder feature -> separate heads)
        self.heads = nn.ModuleList([
            MLP(
                input_neurons=self.linear_size,
                output_neurons=1,
                hidden_layers_neuron_number=MLP_hidden_layers_neuron_number,
                dropout=dropout_MLP,
                activation_function=self.activation_function
            )
            for _ in range(self.num_horizons)
        ])

    def forward(self, x):
        feats = self.CNN(x)        # [B, C, H', W']
        feats = self.context(feats) # add receptive field, keep size
        feats = self.gap(feats)     # [B, C]

        # Per-horizon outputs
        outs = [head(feats) for head in self.heads]   # list of [B, 1]
        return torch.cat(outs, dim=1)                 # [B, num_horizons]


class Encoder(nn.Module):
    def __init__(self, input_channels, num_layers,
                 output_channels_first_layer=16,
                 channels_increase_per_layer=2,
                 max_channels=256,                 # <--- new
                 use_residuals=True,
                 downsample_mode="maxpool",
                 conv_multiple='single',
                 use_bn=True,
                 p_dropout=0.2,
                 activation_function='ReLU'):
        super().__init__()
        self.num_layers = num_layers

        prev_out_ch = input_channels
        for layer in range(num_layers):
            out_ch = int(output_channels_first_layer * (channels_increase_per_layer ** layer))
            if max_channels is not None:
                out_ch = min(out_ch, max_channels)

            Block = ResidualDownBlock if use_residuals else DownBlock
            module = Block(
                prev_out_ch, out_ch,
                conv_multiple=conv_multiple,
                down_mode=downsample_mode,
                act=activation_function,
                use_bn=use_bn,
                p_drop=p_dropout,
                conv_down_kernel=2 if downsample_mode == "doubleconv" else 3
            )
            setattr(self, f"convblock{layer}", module)

            prev_out_ch = out_ch
            
    def forward(self, x):
        for layer in range(self.num_layers):
            x = getattr(self, f"convblock{layer}")(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, input_channels, num_layers,
#                  output_channels_first_layer=16,
#                  channels_increase_per_layer=2,
#                  use_residuals=True,           # bool or list[bool]
#                  downsample_mode="maxpool",    # str or list[str]
#                  conv_multiple='single',       # 'single' or 'double'
#                  use_bn = True,
#                  p_dropout=0.2, 
#                  activation_function='ReLU'):
#         super().__init__()
#         self.num_layers = num_layers
        
#         for layer in range(num_layers):
            
#             in_ch  = output_channels_first_layer * (channels_increase_per_layer ** (layer-1)) if layer > 0 else input_channels
#             out_ch = output_channels_first_layer * (channels_increase_per_layer **  layer)
#             out_ch = min(out_ch, 256)
#             in_ch = min(out_ch, 256)
#             # map legacy names to unified modes
#             Block = ResidualDownBlock if use_residuals else DownBlock
#             module = Block(in_ch, out_ch,
#                            conv_multiple=conv_multiple,
#                            down_mode=downsample_mode,
#                            act=activation_function,
#                            use_bn=use_bn,
#                            p_drop=p_dropout,
#                            conv_down_kernel=2 if downsample_mode == "doubleconv" else 3)
#             setattr(self, f"convblock{layer}", module)

#     def forward(self, x):
#         for layer in range(self.num_layers):
#             x = getattr(self, f"convblock{layer}")(x)
#         return x

# class Encoder(nn.Module):
#     def __init__(self, input_channels, num_layers,
#                  output_channels_first_layer=16,
#                  channels_increase_per_layer=2,
#                  use_residuals = True, # can be list
#                  downsample_mode="conv+pool", # Can be a list
#                  global_kwargs=dict(),
#                  kwargs_per_layer = [],
#                  activation_function = 'ReLU'):
#         super().__init__()
#         self.activation_function = activation_function
#         self.num_layers = num_layers
#         if type(downsample_mode)==str:
#             downsample_modes = [downsample_mode for _ in range(self.num_layers)]
#         if not isinstance(use_residuals, list):
#             use_residuals = [use_residuals for _ in range(self.num_layers)]
#         if not global_kwargs:
#             if not kwargs_per_layer:
#                 kwargs_per_layer = [dict() for _ in range(self.num_layers)]
#         else:
#             kwargs_per_layer = [global_kwargs for _ in range(self.num_layers)]
#         for layer in range(self.num_layers):
#             in_channels_layer = output_channels_first_layer*(channels_increase_per_layer**(layer-1)) if layer>0 else input_channels
#             out_channels_layer = output_channels_first_layer*(channels_increase_per_layer**(layer)) if layer>0 else output_channels_first_layer
#             if use_residuals: 
#                 module =  ResidualBlock(ConvolutionBlock, in_channels_layer, out_channels_layer,downsample_mode = downsample_modes[layer], residual_activation_function=self.activation_function,
#                                         activation_function=self.activation_function, **kwargs_per_layer[layer])
#             else:
#                 module =  ConvolutionBlock(in_channels_layer, out_channels_layer,downsample_modes[layer], activation_function=self.activation_function,
#                                            **kwargs_per_layer[layer])

#             setattr(self, f"convblock{layer}", module)
#     def forward(self, x):
#         out = x
#         for layer in range(self.num_layers):
#             out = getattr(self, f"convblock{layer}")(out)
#         return out



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