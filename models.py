import torch
import torch.nn as nn

class TaggerResNet(nn.Module):
    def __init__(self, num_tags: int):
        super(TaggerResNet, self).__init__()
        
        # Determine the depth based on num_tags
        if num_tags <= 10:
            num_blocks = 2
        elif num_tags <= 50:
            num_blocks = 3
        elif num_tags <= 100:
            num_blocks = 4
        else:
            num_blocks = 5
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Intermediate layers determined by the number of blocks
        self.layer1 = self._make_layer(64, 128, num_blocks)
        self.layer2 = self._make_layer(128, 256, num_blocks)
        self.layer3 = self._make_layer(256, 512, num_blocks)
        self.layer4 = self._make_layer(512, 1024, num_blocks)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully connected layer
        self.fc = nn.Linear(1024, num_tags)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
