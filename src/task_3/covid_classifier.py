import torch.nn as nn
import torch.nn.functional as F

class CovidNet(nn.Module):
    def __init__(self):
        super(CovidNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(16, 3)

    def forward(self, x):
        x = self.network(x)
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x
