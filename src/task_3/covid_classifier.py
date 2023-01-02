import torch.nn as nn
import torch.nn.functional as F


class CovidNet(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.fc1 = nn.Linear(196608, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 3)
        # Dropout Value
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = F.log_softmax(self.fc6(x), dim=1)
        return x
