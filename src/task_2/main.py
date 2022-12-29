import struct

import numpy as np

# Adapted from https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
# Converts the MNIST .idx files to useable numpy arrays
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist(location, tr_x_fn, tr_y_fn, te_x_fn, te_y_fn):
    train_x = read_idx(location + tr_x_fn)
    train_y = read_idx(location + tr_y_fn)
    test_x = read_idx(location + te_x_fn)
    test_y = read_idx(location + te_y_fn)
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_mnist("../../data/MNIST/", "train-images.idx3-ubyte",
                                                  "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                                  "t10k-labels.idx1-ubyte")
