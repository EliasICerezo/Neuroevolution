from neuroevolution.basic_neural_network import BasicNeuralNetwork
import numpy as np

if __name__ == "__main__":
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    weights = np.random.rand(3,1)
    bias = np.random.rand(1)
    # print(output_layer)
    #Creating the actual nnet
    nnet = BasicNeuralNetwork(weights=weights, biases=bias)
    nnet.train(feature_set,labels, 1)
