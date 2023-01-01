import struct
import sys

import numpy as np
from matplotlib import pyplot as plt
import pickle
from src.task_2.cnn import My_Cnn


# Adapted from https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
# Converts the MNIST .idx files to usable numpy arrays
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# loads MNIST dataset and shuffles the testing and training set according to a seed
# converts the image data from 28x28 to 784x1 and divides it by 255 (maximum value)
def load_mnist(location, tr_x_fn, tr_y_fn, te_x_fn, te_y_fn, seed):
    tr_x_raw = np.array(read_idx(location + tr_x_fn), copy=True)
    tr_x = []
    for n in range(0, len(tr_x_raw)):
        tr_x.append(np.reshape(tr_x_raw[n], tr_x_raw[n].size))
    tr_x = np.array(tr_x)
    tr_x = tr_x.T / 255
    tr_y = read_idx(location + tr_y_fn)
    te_x_raw = np.array(read_idx(location + te_x_fn), copy=True)
    te_x = []
    for n in range(0, len(te_x_raw)):
        te_x.append(np.reshape(te_x_raw[n], te_x_raw[n].size))
    te_x = np.array(te_x)
    te_x = te_x.T / 255
    te_y = read_idx(location + te_y_fn)
    print("Data Loaded!")
    print("Train set size: " + str(len(tr_x[0])))
    print("Test set size: " + str(len(te_x[0])))
    return tr_x, tr_y, te_x, te_y

# Plots a graph of the sigmoid function of the cnn
def plot_sigmoid():
    x_ax = np.arange(-5, 5, 0.1)
    y_ax = []
    for n in x_ax:
        y_ax.append(My_Cnn.sigmoid(n))
    plt.plot(x_ax, y_ax)
    plt.xlabel("Value")
    plt.ylabel("Sigmoid of value")
    plt.show()

# Plots a graph of the inverse sigmoid function of the cnn
def plot_sigmoid_back():
    x_ax = np.arange(-5, 5, 0.1)
    y_ax = []
    for n in x_ax:
        y_ax.append(My_Cnn.sigmoid_back(n))
    plt.plot(x_ax, y_ax)
    plt.xlabel("Value")
    plt.ylabel("Derivative Sigmoid of value")
    plt.show()

# Plots a graph of the ReLU function of the cnn
def plot_relu():
    x_ax = np.arange(-5, 5, 0.1)
    y_ax = []
    for n in x_ax:
        y_ax.append(My_Cnn.relu(n))
    plt.plot(x_ax, y_ax)
    plt.xlabel("Value")
    plt.ylabel("Derivative ReLU of value")
    plt.show()

# Plots a graph of the inverse ReLU function of the cnn
def plot_relu_back():
    x_ax = np.arange(-5, 5, 0.1)
    y_ax = []
    for n in x_ax:
        y_ax.append(My_Cnn.relu_back(n))
    plt.plot(x_ax, y_ax)
    plt.xlabel("Value")
    plt.ylabel("ReLU of value")
    plt.show()

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    plot_sigmoid_back()
    # random_seed = np.random.randint(0, 10000)
    # train_x, train_y, test_x, test_y = load_mnist("../../data/MNIST/", "train-images.idx3-ubyte",
    #                                               "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
    #                                               "t10k-labels.idx1-ubyte", random_seed)
    #
    # cnn = My_Cnn([["relu", 10], ["sigmoid", 10]], len(train_x), 10)
    # predictions = cnn.train(train_x, train_y, 0.2, 10000, 0.8, True)
    #
    # save_model(cnn, 'models/relu_10_sig_10_with_drop_0-8_alpha_0-2_iter_10000')
    #
    # predictions_array = np.zeros((len(predictions), 2))
    # for i in range(0, len(predictions)):
    #     predictions_array[i] = predictions[i]
    #     predictions_array[i] = predictions[i]
    # plt.plot(predictions_array[:, 1], predictions_array[:, 0])
    # plt.xlabel("Iterations")
    # plt.ylabel("Prediction Accuracy")
    # plt.show()
