import torch.nn as nn

class AdulterantDetector(nn.Module):

    def __init__(self, convpool_layers, fc_layers, encoding_dim=10):
        super(AdulterantDetector, self).__init__()

        self.encoding_dim = encoding_dim

        self.layers = []
        for i in range(len(convpool_layers)-1):
            self.layers.append(nn.Conv2d(convpool_layers[i], convpool_layers[i+1], kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Flatten())

        for i in range(len(fc_layers) - 2):
            self.layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_layers[-2], fc_layers[-1]))

        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.sequential(x)
        return out