import numpy as np

from src.task_2.cnn import My_Cnn

def test_cnn():
    # Checks that the relu layers outputs correctly
    assert My_Cnn.relu(1) == 1
    assert My_Cnn.relu(-1) == 0
    assert My_Cnn.relu(-17) == 0
    assert My_Cnn.relu(0) == 0
    assert My_Cnn.relu(1098) == 1098
    # Checks that the relu back layers outputs correctly
    assert My_Cnn.relu_back(1) == 1
    assert My_Cnn.relu_back(-1) == 0
    assert My_Cnn.relu_back(0) == 0
    assert My_Cnn.relu_back(100) == 1
    # Checks that the sigmoid layer outputs correctly
    # Values taken from https://keisan.casio.com/exec/system/15157249643325
    assert My_Cnn.sigmoid(1) == 0.7310585786300048792512
    assert My_Cnn.sigmoid(0) == 0.5
    assert My_Cnn.sigmoid(-1) == 0.268941421369995121
    # Checks that the sigmoid back layer outputs correctly
    # Values taken from https://keisan.casio.com/exec/system/15157249643425
    assert My_Cnn.sigmoid_back(1) == 0.1966119332414818525374
    assert My_Cnn.sigmoid_back(0) == 0.2350037122015944890693
    assert My_Cnn.sigmoid_back(-1) == 0.1966119332414818525374



if __name__ == '__main__':
    test_cnn()
