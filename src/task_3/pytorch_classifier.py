import torch.nn as nn
import torch.nn.functional as F

# Adapted from # Adapted from https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4
class CNN(nn.Module):
    # initialises the model
    def __init__(self, dropout):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        # Dropout Value
        self.dropout = nn.Dropout(dropout)

    # runs the forward pass on the model
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
