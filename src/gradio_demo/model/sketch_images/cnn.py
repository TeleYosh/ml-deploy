import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_filters, hidden_dim, n_layers, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_filters, 2*n_filters, 5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.input_dim = 960
        self.flatten = nn.Flatten()
        self.inp_layer = nn.Linear(self.input_dim, hidden_dim)
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3)
            ) for i in range(n_layers)
        ])
        self.out_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.inp_layer(torch.flatten(x, start_dim=1))
        for layer in self.classifier:
            x = layer(x)
        x = self.out_layer(x)
        return x