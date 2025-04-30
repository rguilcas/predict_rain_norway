from torch import nn
import torch


class CNN_Model(nn.Module):
    def __init__(self, 
                 input_channels, 
                 image_size, 
                 num_classes,
                 number_conv_layers=3,
                 size_conv_kernel=3,
                 out_channels_conv1=12, 
                 out_channel_factor_increase_per_layer = 2,
                 ):
        super(CNN_Model, self).__init__()
        
        # CNN layers to extract spatial features
        self.num_classes = num_classes
        self.conv_layer1 = nn.Sequential(
                                nn.Conv2d(input_channels, 
                                          out_channels_conv1, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer2 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer3 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer4 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**3, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        
        if number_conv_layers == 1:
            self.cnn = nn.Sequential(self.conv_layer1)
        elif number_conv_layers == 2:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2)
        elif number_conv_layers == 3:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3)
        elif number_conv_layers == 4:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4)
        
        max_pools = 4**(number_conv_layers)
        self.output_cnn_channels  = out_channels_conv1*out_channel_factor_increase_per_layer**(number_conv_layers-1)
        self.linear_size = image_size * self.output_cnn_channels // max_pools 
        self.fc = nn.Linear(self.linear_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        # Apply CNN to extract spatial features
        batch_size,  channels, height, width = x.size()
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch_size, -1)  # Shape: (batch_size, seq_length, features)
        output = self.fc(cnn_out)  # Shape: (batch_size, output_size)
        # output = torch.sigmoid(output) 
        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global context
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNN_Model_ChannelAttention(nn.Module):
    def __init__(self, 
                 input_channels, 
                 image_size, 
                 num_classes,
                 number_conv_layers=3,
                 size_conv_kernel=3,
                 out_channels_conv1=12, 
                 out_channel_factor_increase_per_layer = 2,
                 ):
        super(CNN_Model_ChannelAttention, self).__init__()
        
        # CNN layers to extract spatial features
        self.num_classes = num_classes
        self.conv_layer1 = nn.Sequential(
                                nn.Conv2d(input_channels, 
                                          out_channels_conv1, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer2 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer3 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer4 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**3, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        
        if number_conv_layers == 1:
            self.cnn = nn.Sequential(self.conv_layer1)
        elif number_conv_layers == 2:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2)
        elif number_conv_layers == 3:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3)
        elif number_conv_layers == 4:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4)
        
        max_pools = 4**(number_conv_layers)
        self.output_cnn_channels  = out_channels_conv1*out_channel_factor_increase_per_layer**(number_conv_layers-1)
        self.attention = ChannelAttention(self.output_cnn_channels)

        self.linear_size = image_size * self.output_cnn_channels // max_pools 
        self.fc = nn.Linear(self.linear_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        # Apply CNN to extract spatial features
        batch_size,  channels, height, width = x.size()
        cnn_out = self.cnn(x)
        cnn_out = self.attention(cnn_out)
        cnn_out = cnn_out.view(batch_size, -1)  # Shape: (batch_size, seq_length, features)
        output = self.fc(cnn_out)  # Shape: (batch_size, output_size)
        # output = torch.sigmoid(output) 
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn_map = self.conv(x_cat)  # (B, 1, H, W)
        return x * attn_map  # Apply attention map



class CNN_Model_SpatialAttention(nn.Module):
    def __init__(self, 
                 input_channels, 
                 image_size, 
                 num_classes,
                 number_conv_layers=3,
                 size_conv_kernel=3,
                 out_channels_conv1=12, 
                 out_channel_factor_increase_per_layer = 2,
                 ):
        super(CNN_Model_SpatialAttention, self).__init__()
        
        # CNN layers to extract spatial features
        self.num_classes = num_classes
        self.conv_layer1 = nn.Sequential(
                                nn.Conv2d(input_channels, 
                                          out_channels_conv1, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer2 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer3 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer4 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**3, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        
        if number_conv_layers == 1:
            self.cnn = nn.Sequential(self.conv_layer1)
        elif number_conv_layers == 2:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2)
        elif number_conv_layers == 3:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3)
        elif number_conv_layers == 4:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4)
        
        max_pools = 4**(number_conv_layers)
        self.output_cnn_channels  = out_channels_conv1*out_channel_factor_increase_per_layer**(number_conv_layers-1)
        self.attention = SpatialAttention()

        self.linear_size = image_size * self.output_cnn_channels // max_pools 
        self.fc = nn.Linear(self.linear_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        # Apply CNN to extract spatial features
        batch_size,  channels, height, width = x.size()
        cnn_out = self.cnn(x)
        cnn_out = self.attention(cnn_out)
        cnn_out = cnn_out.view(batch_size, -1)  # Shape: (batch_size, seq_length, features)
        output = self.fc(cnn_out)  # Shape: (batch_size, output_size)
        # output = torch.sigmoid(output) 
        return output



class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    
class CNN_Model_CBAM(nn.Module):
    def __init__(self, 
                 input_channels, 
                 image_size, 
                 num_classes,
                 number_conv_layers=3,
                 size_conv_kernel=3,
                 out_channels_conv1=12, 
                 out_channel_factor_increase_per_layer = 2,
                 ):
        super(CNN_Model_CBAM, self).__init__()
        
        # CNN layers to extract spatial features
        self.num_classes = num_classes
        self.conv_layer1 = nn.Sequential(
                                nn.Conv2d(input_channels, 
                                          out_channels_conv1, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer2 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer3 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer4 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**3, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        
        if number_conv_layers == 1:
            self.cnn = nn.Sequential(self.conv_layer1)
        elif number_conv_layers == 2:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2)
        elif number_conv_layers == 3:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3)
        elif number_conv_layers == 4:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4)
        
        max_pools = 4**(number_conv_layers)
        self.output_cnn_channels  = out_channels_conv1*out_channel_factor_increase_per_layer**(number_conv_layers-1)
        self.attention = CBAM(self.output_cnn_channels)

        self.linear_size = image_size * self.output_cnn_channels // max_pools 
        self.fc = nn.Linear(self.linear_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        # Apply CNN to extract spatial features
        batch_size,  channels, height, width = x.size()
        cnn_out = self.cnn(x)
        cnn_out = self.attention(cnn_out)
        cnn_out = cnn_out.view(batch_size, -1)  # Shape: (batch_size, seq_length, features)
        output = self.fc(cnn_out)  # Shape: (batch_size, output_size)
        # output = torch.sigmoid(output) 
        return output

class CNN_3dModel(nn.Module):
    def __init__(self, 
                 input_channels, 
                 image_size, 
                 num_classes,
                 number_conv_layers=3,
                 size_conv_kernel=3,
                 out_channels_conv1=12, 
                 out_channel_factor_increase_per_layer = 2,
                 ):
        super(CNN_Model, self).__init__()
        
        # CNN layers to extract spatial features
        self.num_classes = num_classes
        self.conv_layer1 = nn.Sequential(
                                nn.Conv2d(input_channels, 
                                          out_channels_conv1, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer2 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          kernel_size=size_conv_kernel, 
                                          stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer3 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer4 = nn.Sequential(
                                nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer**2, 
                                          out_channels_conv1*out_channel_factor_increase_per_layer**3, 
                                          kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        
        if number_conv_layers == 1:
            self.cnn = nn.Sequential(self.conv_layer1)
        elif number_conv_layers == 2:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2)
        elif number_conv_layers == 3:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3)
        elif number_conv_layers == 4:
            self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4)
        
        max_pools = 4**(number_conv_layers)
        self.output_cnn_channels  = out_channels_conv1*out_channel_factor_increase_per_layer**(number_conv_layers-1)
        self.linear_size = image_size * self.output_cnn_channels // max_pools 
        self.fc = nn.Linear(self.linear_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        # Apply CNN to extract spatial features
        batch_size,  channels, height, width = x.size()
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch_size, -1)  # Shape: (batch_size, seq_length, features)
        output = self.fc(cnn_out)  # Shape: (batch_size, output_size)
        output = torch.sigmoid(output) 
        return output
# def get_model(wandb_logger):
#     model_name = wandb_logger.experiment.config['model_choice']
#     if model_name == 'CNN':
#         NN = Wang2024(
#             num_classes=wandb_logger.experiment.config['num_classes'],
#             input_channels=wandb_logger.experiment.config['num_channels']*wandb_logger.experiment.config['num_days_predictant'], 
#             image_size=wandb_logger.experiment.config['image_size'], 
#             number_conv_layers=wandb_logger.experiment.config['num_conv_layer'],
#             size_conv_kernel=wandb_logger.experiment.config['size_conv_kernel'],
#             out_channels_conv1=wandb_logger.experiment.config['conv1_kernel_number'], 
#             out_channel_factor_increase_per_layer =wandb_logger.experiment.config['factor_increase_kernels_per_conv_layer'],
#             dropout_proba=0
#             )
        
#     elif model_name == 'CNN_LSTM':
#         NN = CNN_LSTM_Model(
#             num_classes=wandb_logger.experiment.config['num_classes'],
#             input_channels=wandb_logger.experiment.config['num_channels'], 
#             image_size=wandb_logger.experiment.config['image_size'], 
#             number_conv_layers=wandb_logger.experiment.config['num_conv_layer'],
#             size_conv_kernel=wandb_logger.experiment.config['size_conv_kernel'],
#             out_channels_conv1=wandb_logger.experiment.config['conv1_kernel_number'], 
#             out_channel_factor_increase_per_layer =wandb_logger.experiment.config['factor_increase_kernels_per_conv_layer'],
#             hidden_size=wandb_logger.experiment.config['LSTM_hidden_size'],)
#     elif model_name == 'CNN_BiLSTM_attention':
#         NN = CNN_BiLSTM_Attention(
#             input_channels=wandb_logger.experiment.config['num_channels'],
#             num_fc_units=256,
#             num_output = wandb_logger.experiment.config['num_days_predicted'],
#             image_size = wandb_logger.experiment.config['image_size'])
#     return NN



# class Wang2024(nn.Module):
#     def __init__(self, 
#                  num_classes,
#                  input_channels, 
#                  image_size, 
#                  number_conv_layers=3,
#                  size_conv_kernel=3,
#                  out_channels_conv1=12, 
#                  out_channel_factor_increase_per_layer = 2,
#                  dropout_proba=0):
#         super().__init__()
#         self.num_classes = num_classes
#         self.conv_layer1 = nn.Sequential(
#                                 nn.Conv2d(input_channels, 
#                                           out_channels_conv1, 
#                                           kernel_size=size_conv_kernel, 
#                                           stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
#         self.conv_layer2 = nn.Sequential(
#                                 nn.Conv2d(out_channels_conv1, 
#                                           out_channels_conv1*out_channel_factor_increase_per_layer, 
#                                           kernel_size=size_conv_kernel, 
#                                           stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
#         self.conv_layer3 = nn.Sequential(
#                                 nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer, 
#                                           out_channels_conv1*out_channel_factor_increase_per_layer**2, 
#                                           kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
#         self.conv_layer4 = nn.Sequential(
#                                 nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer**2, 
#                                           out_channels_conv1*out_channel_factor_increase_per_layer**3, 
#                                           kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
        
#         if number_conv_layers == 1:
#             self.cnn = nn.Sequential(self.conv_layer1)
#         elif number_conv_layers == 2:
#             self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2)
#         elif number_conv_layers == 3:
#             self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3)
#         elif number_conv_layers == 4:
#             self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4)
        
#         max_pools = 4**(number_conv_layers)
#         self.output_cnn_channels  = out_channels_conv1*out_channel_factor_increase_per_layer**(number_conv_layers-1)
#         self.linear_size = image_size * self.output_cnn_channels // max_pools 
#         self.fc1 = nn.Linear(self.linear_size, 96)
#         self.fc2 = nn.Linear(96, 48)
#         self.fc3 = nn.Linear(48, num_classes)

#     def forward(self, x):
#         batch_size, seq_len, c, h, w = x.shape 
#         # reshape tensor to concat time and var_names
#         x = x.view(batch_size, seq_len*c, h, w)
#         x = self.cnn(x)
#         x = x.view((-1,self.linear_size))
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#         x = nn.ReLU()(x)
#         x = self.fc3(x)        
#         x = nn.functional.leaky_relu(x, negative_slope=0.01)
#         x = nn.functional.softplus(x)
#         return x
    

# class CNN_LSTM_Model(nn.Module):
#     def __init__(self, 
#                  input_channels, 
#                  image_size, 
#                  num_classes,
#                  number_conv_layers=3,
#                  size_conv_kernel=3,
#                  out_channels_conv1=12, 
#                  out_channel_factor_increase_per_layer = 2,
#                  hidden_size=256, 
#                  ):
#         super(CNN_LSTM_Model, self).__init__()
        
#         # CNN layers to extract spatial features
#         self.num_classes = num_classes
#         self.conv_layer1 = nn.Sequential(
#                                 nn.Conv2d(input_channels, 
#                                           out_channels_conv1, 
#                                           kernel_size=size_conv_kernel, 
#                                           stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
#         self.conv_layer2 = nn.Sequential(
#                                 nn.Conv2d(out_channels_conv1, 
#                                           out_channels_conv1*out_channel_factor_increase_per_layer, 
#                                           kernel_size=size_conv_kernel, 
#                                           stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
#         self.conv_layer3 = nn.Sequential(
#                                 nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer, 
#                                           out_channels_conv1*out_channel_factor_increase_per_layer**2, 
#                                           kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
#         self.conv_layer4 = nn.Sequential(
#                                 nn.Conv2d(out_channels_conv1*out_channel_factor_increase_per_layer**2, 
#                                           out_channels_conv1*out_channel_factor_increase_per_layer**3, 
#                                           kernel_size=size_conv_kernel, stride=1, padding=(size_conv_kernel-1)//2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2))
        
#         if number_conv_layers == 1:
#             self.cnn = nn.Sequential(self.conv_layer1)
#         elif number_conv_layers == 2:
#             self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2)
#         elif number_conv_layers == 3:
#             self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3)
#         elif number_conv_layers == 4:
#             self.cnn = nn.Sequential(self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4)
        
#         max_pools = 4**(number_conv_layers)
#         self.output_cnn_channels  = out_channels_conv1*out_channel_factor_increase_per_layer**(number_conv_layers-1)
#         self.linear_size = image_size * self.output_cnn_channels // max_pools 
#         # ouptut_size_cnn
#         # LSTM layer to capture temporal dependencies
#         self.lstm = nn.LSTM(input_size=self.linear_size, hidden_size=hidden_size, batch_first=True)
        
#         # Fully connected layer for final prediction
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         # x shape: (batch_size, seq_length, channels, height, width)
#         # Apply CNN to extract spatial features
#         batch_size, seq_length, channels, height, width = x.size()
#         cnn_out = []
#         for i in range(seq_length):
#             cnn_out.append(self.cnn(x[:, i, :, :, :]))  # Extract features from each time step
#         cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, seq_length, channels, height, width)
#         # Flatten CNN output for LSTM
#         cnn_out = cnn_out.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, features)
#         # Apply LSTM
#         lstm_out, _ = self.lstm(cnn_out)
        
#         # Get last time step output and pass through fully connected layer
#         lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
#         output = self.fc(lstm_out)  # Shape: (batch_size, output_size)
#         output = nn.functional.leaky_relu(output, negative_slope=0.01)
#         output = nn.functional.softplus(output)
#         return output


# class Attention(nn.Module):
#     """Self-Attention Mechanism to focus on important timesteps."""
#     def __init__(self, input_dim):
#         super(Attention, self).__init__()
#         self.attn = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         # x: (batch, seq_len, hidden_dim)
#         attn_weights = torch.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
#         attended_output = torch.sum(attn_weights * x, dim=1)  # Weighted sum over timesteps
#         return attended_output, attn_weights

# class CNN_BiLSTM_Attention(nn.Module):
#     def __init__(self, input_channels=3, num_lstm_units=128, num_fc_units=64,  num_output=1, image_size=128*64):
#         super(CNN_BiLSTM_Attention, self).__init__()

#         # CNN for spatial feature extraction
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )

        
#         # ouptut_size_cnn
#         self.lin_out_cnn_size = 64 * image_size // 4 // 4
#         # BiLSTM for temporal learning
#         self.bilstm = nn.LSTM(
#             input_size=self.lin_out_cnn_size,
#             hidden_size=num_lstm_units,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         # Attention mechanism
#         self.attention = Attention(input_dim=num_lstm_units * 2)  # BiLSTM is bidirectional

#         # Fully connected layer for regression
#         self.fc = nn.Linear(num_lstm_units * 2, num_fc_units)
#         self.output_layer = nn.Linear(num_fc_units, num_output)  # Output: Rainfall amount

#         self.criterion = nn.MSELoss()

#     def forward(self, x):
#         batch_size, seq_len, c, h, w = x.shape  # (batch, seq_len, channels, height, width)
#         cnn_features = []
#         for t in range(seq_len):
#             out = self.cnn(x[:, t, :, :, :])  # (batch, 64, H/4, W/4)
#             out = out.view(batch_size, -1)  # Flatten
#             cnn_features.append(out)

#         cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, feature_dim)

#         # BiLSTM processing
#         lstm_out, _ = self.bilstm(cnn_features)  # (batch, seq_len, 2 * hidden_size)

#         # Apply Attention
#         attended_out, attn_weights = self.attention(lstm_out)  # (batch, hidden_dim)

#         # Fully connected layers
#         x = self.fc(attended_out)
#         x = self.output_layer(x)  # (batch, 1)

#         return x









# class LeNET(nn.Module):
#     def __init__(self, num_classes,num_channels_in, image_size, groups=3):
        
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=12, kernel_size=3, stride=1, padding=1, groups=groups)
#         self.bn1 = nn.BatchNorm2d(12)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(24)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         cnn_channels = 24
#         max_pools = 4*4
#         self.linear_size = image_size * cnn_channels // max_pools 
#         self.fc1 = nn.Linear(self.linear_size, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = nn.ReLU()(x)
#         x = self.pool1(x)

#         x = nn.Dropout()(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = nn.ReLU()(x)
#         x = self.pool2(x)

#         x = x.view((-1,self.linear_size))
#         x = nn.ReLU()(self.fc1(x))
#         x = self.fc2(x)
#         return nn.ReLU()(x)