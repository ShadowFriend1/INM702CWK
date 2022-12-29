import numpy as np

from src.task_2.main import load_mnist

def test_load():
    train_x, train_y, test_x, test_y = load_mnist("../../data/MNIST/", "train-images.idx3-ubyte",
                                                  "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                                  "t10k-labels.idx1-ubyte")
    # Checks that all elements of the resulting arrays are the same size
    same_size = True
    current_check = train_x[0]
    for n in train_x:
        if n.size != current_check.size:
            same_size = False
            break
        current_check = n
    assert same_size


if __name__ == '__main__':
    test_load()
