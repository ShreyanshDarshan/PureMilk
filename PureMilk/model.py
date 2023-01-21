import torch.nn as nn

class AdulterantDetector(nn.Module):

    def __init__(self, layers, encoding_dim=10):
        super(AdulterantDetector, self).__init__()

        if layers[0] % 2 * encoding_dim != 0:
            raise ValueError("Input size must be a multiple of 2*encoding_dim")
        self.encoding_dim = encoding_dim

        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.sequential(x)
        return out