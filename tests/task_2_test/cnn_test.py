import numpy as np

from src.task_2.cnn import My_Cnn

def test_cnn():
    # Checks that the relu layers outputs correctly
    assert My_Cnn.relu(1) == 1
    assert My_Cnn.relu(-1) == 0
    assert My_Cnn.relu(0) == 0
    # Checks that the sigmoid layer outputs correctly
    assert My_Cnn.sigmoid(1) == 0.7310585786300049
    assert My_Cnn.sigmoid_back(1) == 0


if __name__ == '__main__':
    test_cnn()
