import struct
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


if __name__ == "__main__":
    random_seed = np.random.randint(0, 10000)
    train_x, train_y, test_x, test_y = load_mnist("../../data/MNIST/", "train-images.idx3-ubyte",
                                                  "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                                  "t10k-labels.idx1-ubyte", random_seed)
    # gradient_descent(train_x, train_y, 0.10, 500)
    cnn = My_Cnn([["relu", 10], ["sigmoid", 10]], 0.7, len(train_x), 10)
    cnn.train(train_x, train_y, 0.2, 500)
    sys.exit(0)
