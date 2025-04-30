from torch import nn, optim

class FullyConnected3LayerBinaryClassifier(nn.Module):
    """
    Simple fully connected model with one hidden layer
    """
    def __init__(self, input_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer with num_classes neurons
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)     
        return x
    
class FullyConnected5LayerBinaryClassifier(nn.Module):
    """
    Simple fully connected model with 3 hidden layer
    """
    def __init__(self, input_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 32)
        self.fc5 = nn.Linear(32, 1)  # Output layer with num_classes neurons
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        x = nn.Sigmoid()(x)     
        return x
    

class CNNBinaryClassifier(nn.Module):
    def __init__(self, num_channels_in, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_size = image_size*16//2//2//4
        self.fc1 = nn.Linear(self.linear_size, 512)
        self.fc2 = nn.Linear(512, 1)
    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = x.view((-1,self.linear_size))
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = nn.Sigmoid()(x)     
        return x
    
class CNNQuantilesClassifier(nn.Module):
    def __init__(self, quantiles,num_channels_in, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=8, kernel_size=3, stride=1, padding=1,)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_size = image_size*16//2//2//4
        self.fc1 = nn.Linear(self.linear_size, 512)
        self.fc2 = nn.Linear(512, quantiles)
    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = nn.Dropout()(x)
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = x.view((-1,self.linear_size))
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNQuantilesClassifierDepthWise(nn.Module):
    def __init__(self, quantiles,num_channels_in, image_size, groups=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=12, kernel_size=3, stride=1, padding=1,)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_size = image_size*4*2//2//2//4*3
        self.fc1 = nn.Linear(self.linear_size, 512)
        self.fc2 = nn.Linear(512, quantiles)
    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = nn.Dropout()(x)
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = x.view((-1,self.linear_size))
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
