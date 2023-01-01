import sys

import numpy as np


# A class representing the cnn
class My_Cnn:
    # Initialises the variables needed by the network
    def __init__(self, layers, data_size, output_classes):
        # layers is a list of what layers will make up the cnn along with how many nodes they will contain
        self.layers = layers

        # Start with random weights for all layers
        weights = []
        for i in range(0, len(layers)):
            if i > 0:
                prev_layer = layers[i - 1][1]
            else:
                prev_layer = data_size
            if layers[i][0] in ("relu", "sigmoid"):
                weights.append([np.random.randn(layers[i][1], prev_layer), np.random.randn(layers[i][1], 1)])
            else:
                print("No " + layers[i][0] + " Layer! During Initialisation")
                sys.exit(1)

        # Setup weights for the output layer
        weights.append([np.random.randn(output_classes, output_classes), np.random.randn(output_classes, 1)])

        self.weights = weights

    # Returns the result of a sigmoid function applied to a value/array after normalising the array to prevent overflow
    # errors (takes a long time)
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Return the result of an inverse sigmoid function applied to a value/array
    @staticmethod
    def sigmoid_back(z):
        return z * (1 - z)

    # Returns the results of a ReLU function applied to a value/array
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    # Returns the results of an inverse ReLU function applied to a value/array
    @staticmethod
    def relu_back(z):
        return z > 0

    # Returns the results of a softmax function applied to a value/array
    @staticmethod
    def softmax(z):
        return np.exp(z) / sum(np.exp(z))

    # Applies dropout to an array, returning the array with dropout applied as well as the mask used
    # (for backwards pass)
    @staticmethod
    def dropout(z, p):
        mask = (np.random.rand(*z.shape) < p) / p
        return mask * z, mask

    # Runs a forward pass through the cnn
    def forward_pass(self, x, p, dropout):
        # Applies dropout with probability p if dropout is true (used for training)
        masks = []
        if dropout:
            x, mask = My_Cnn.dropout(x, p)
            masks.append(mask)

        z_layers = []
        a_layers = []

        # Pass through all layers
        for i in range(0, len(self.layers)):
            if i > 0:
                prev_a = a_layers[-1]
            else:
                prev_a = x
            z = self.weights[i][0].dot(prev_a) + self.weights[i][1]
            if dropout:
                z, mask = My_Cnn.dropout(z, p)
                masks.append(mask)
            z_layers.append(z)
            if self.layers[i][0] == "relu":
                a = My_Cnn.relu(z_layers[i])
                a_layers.append(a)
            elif self.layers[i][0] == "sigmoid":
                a = My_Cnn.sigmoid(z_layers[i])
                a_layers.append(a)
            else:
                print("No " + self.layers[i][0] + " Layer! During Forward Pass")
                sys.exit(1)

        # Pass through softmax layer, doesn't apply dropout
        z_layers.append(self.weights[-1][0].dot(a_layers[-1]) + self.weights[-1][1])
        a_layers.append(My_Cnn.softmax(z_layers[-1]))

        return z_layers, a_layers, masks

    # Takes in a result set and reorders it into an array of set size rows and n columns where each column represents a
    # classification of the data (where 1 means true and 0 means false)
    @staticmethod
    def one_hot(y):
        one_hot_y = np.zeros((y.size, y.max() + 1))
        one_hot_y[np.arange(y.size), y] = 1
        one_hot_y = one_hot_y.T
        return one_hot_y

    # Runs a backwards pass through the cnn
    def backward_pass(self, x, y, z_layers, a_layers, masks):
        m = y.size
        one_hot_y = My_Cnn.one_hot(y)
        d_w_layers = []
        d_b_layers = []

        # Backward pass through softmax layer
        dz_f = (a_layers[-1] - one_hot_y)
        d_w_layers.append(1 / m * dz_f.dot(a_layers[-2].T))
        d_b_layers.append(1 / m * np.sum(dz_f))

        prev_dz = dz_f

        # Backward pass through rest of layers
        for i in range(len(self.layers) - 1, -1, -1):
            if i > 0:
                prev_a = a_layers[i - 1]
            else:
                prev_a = x
            if self.layers[i][0] == "relu":
                dz = self.weights[-1][0].T.dot(prev_dz) * My_Cnn.relu_back(z_layers[i])
                if masks:
                    dz *= masks[i + 1]
                prev_dz = dz
            elif self.layers[i][0] == "sigmoid":
                dz = self.weights[-1][0].T.dot(prev_dz) * My_Cnn.sigmoid_back(z_layers[i])
                if masks:
                    dz *= masks[i + 1]
                prev_dz = dz
            else:
                print("No " + self.layers[i][0] + " Layer! During Backward Pass")
                sys.exit(1)
            d_w_layers.append(1 / m * dz.dot(prev_a.T))
            d_b_layers.append(1 / m * np.sum(dz))
        d_w_layers.reverse()
        d_b_layers.reverse()
        return d_w_layers, d_b_layers

    # Updates the network layer weights
    def update_weights(self, d_w_layers, d_b_layers, alpha):
        for i in range(0, len(self.weights)):
            self.weights[i][0] -= alpha * d_w_layers[i]
            self.weights[i][1] -= alpha * d_b_layers[i]

    @staticmethod
    def get_predictions(a_f):
        return np.argmax(a_f, 0)

    @staticmethod
    def get_accuracy(predictions, y):
        return np.sum(predictions == y) / y.size

    def train(self, x, y, alpha, iterations, dropout_p, dropout):
        predictions = []
        for i in range(0, iterations):
            z_layers, a_layers, mask = self.forward_pass(x, dropout_p, dropout)
            d_w_layers, d_b_layers = self.backward_pass(x, y, z_layers, a_layers, mask)
            self.update_weights(d_w_layers, d_b_layers, alpha)
            if i % 10 == 0:
                print("iteration: ", i)
                prediction = My_Cnn.get_predictions(a_layers[-1])
                predictions.append([My_Cnn.get_accuracy(prediction, y), i])
                print(predictions[-1][0])
            final_i = i
        prediction = My_Cnn.get_predictions(a_layers[-1])
        predictions.append([My_Cnn.get_accuracy(prediction, y), final_i])
        return predictions
