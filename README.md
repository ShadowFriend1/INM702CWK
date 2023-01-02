For ease I have been running my scripts by adding the relevant functions to the python script itself.

Training of networks are left in comments in the main.py file associated with the task.

The trained models are contained in a folder named models within the relevant task folder

Task 1:

To plot the graph of deviations, run plot_deviation()

To plot the graph of shortest path lengths for game sizes run distribution_graphs()

Dijkstra's algorithm is implemented as dijkstra_shortest()

My algorithm is implemented as my_find_shortest()

Task 2:

To load a model from file use load_model()

To run a prediction test for 4 randomised images run prediction_test() (requires a model tyo be loaded from a model file)

Task 3:

load_covid() and load_fashion() load the relevant datasets

train_model() is used to train models

model_test() return accuracy values for the models

both models use the same function to train and test, these functions accept the cnn as an argument