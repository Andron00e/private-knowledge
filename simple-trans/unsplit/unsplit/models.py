import collections

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=1024, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()] + 
            [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] + 
            [nn.Linear(hidden_dim, 10)]
        )
    
    def forward(self, x, start=0, end=7):
        for i, layer in enumerate(self.layers):
            if start <= i <= end:
                x = layer(x)
        return x

class CNN(nn.Module):
    def __init__(self, n_channels=1):
        super(CNN, self).__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=8,
            kernel_size=5
        )
        self.features.append(self.conv1)
        self.layers['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(False)
        self.features.append(self.ReLU1)
        self.layers['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)
        self.layers['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5
        )
        self.features.append(self.conv2)
        self.layers['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(False)
        self.features.append(self.ReLU2)
        self.layers['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)
        self.layers['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layers['fc1'] = self.fc1
     
        self.fc1act = nn.ReLU(False)
        self.classifier.append(self.fc1act)
        self.layers['fc1act'] = self.fc1act
     
        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layers['fc2'] = self.fc2
     
        self.fc2act = nn.ReLU(False)
        self.classifier.append(self.fc2act)
        self.layers['fc2act'] = self.fc2act
     
        self.fc3 = nn.Linear(84, 10)
        self.classifier.append(self.fc3)
        self.layers['fc3'] = self.fc3
        
        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def forward(self, x, start=0, end=10):
        if start <= 5: # start in self.features
            for idx, layer in enumerate(self.features[start:]):
                x = layer(x)
                if idx == end:
                    return x
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + 6 == end:
                    return x
        else:
            if start == 6:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - 6:
                    x = layer(x)
                if idx + 6 == end:
                    return x
                
    def get_params(self, end=10):
        params = []
        for layer in list(self.layers.values())[:end+1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self):
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial.requires_grad_(True)