import numpy as np


class My_Cnn:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_back(da, z):
        sig = My_Cnn.sigmoid(z)
        return da * sig * (1 - sig)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_back(da, z):
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
        return dz

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.exp(z).sum()

    @staticmethod
    def dropout_mask(z, p):
        mask = (np.random.uniform(low=0, high=1, size=z.shape) < p) / p
        print(z.shape, mask.shape)
        return mask

    @staticmethod
    def dropout(z, mask):
        print((mask * z).shape)
        return mask * z

    def forward_pass(self, layers):
        return layers, self.train_x

    def backward_pass(self, layers):
        return layers, self.train_x
