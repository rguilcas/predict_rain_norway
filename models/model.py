from torch import nn, optim


class LeNET(nn.Module):
    def __init__(self, quantiles,num_channels_in, image_size, groups=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=12, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_channels = 24
        max_pools = 4*4
        self.linear_size = image_size * cnn_channels // max_pools 
        self.fc1 = nn.Linear(self.linear_size, 128)
        self.fc2 = nn.Linear(128, quantiles)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = nn.Dropout()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = x.view((-1,self.linear_size))
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return nn.ReLU()(x)
    
class Wang2024(nn.Module):
    def __init__(self, num_classes,num_channels_in, image_size, groups=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=12, kernel_size=3, stride=1, padding=1, groups=groups)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=3, stride=1, padding=1, groups=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_channels = 48
        max_pools = 4*4
        self.linear_size = image_size * cnn_channels // max_pools 
        self.fc1 = nn.Linear(self.linear_size, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = x.view((-1,self.linear_size))
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)        
        return x