import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(
        self,
        n_filters,
        hidden_dim,
        n_layers,
        n_classes,
        input_shape=(1, 28, 28),
        dropout_rate=0,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, 3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_filters, 2 * n_filters, 3, padding=1),
            nn.BatchNorm2d(2 * n_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * n_filters, 4 * n_filters, 3, padding=1),
            nn.BatchNorm2d(4 * n_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_flat_size(dummy_input)
        self.input_dim = conv_out_size
        self.flatten = nn.Flatten()
        self.inp_layer = nn.Linear(self.input_dim, hidden_dim)
        self.classifier = nn.ModuleList()
        for _ in range(n_layers):
            self.classifier.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                )
            )
        self.out_layer = nn.Linear(hidden_dim, n_classes)

    def _get_flat_size(self, x):
        """Helper class to get the flat size of the input after convolutions"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.inp_layer(torch.flatten(x, start_dim=1))
        for layer in self.classifier:
            x = layer(x)
        x = self.out_layer(x)
        return x
