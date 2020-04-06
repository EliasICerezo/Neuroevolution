from neuroevolution.basic_neural_network import BasicNeuralNetwork
import numpy as np

if __name__ == "__main__":
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    weights = np.random.rand(3,2)
    bias = np.random.rand(2)
    # print(output_layer)
    #Creating the actual nnet
    nnet = BasicNeuralNetwork(layers=[3,2,2,2,1], input_size=3, num_of_classes= 1)
    print(nnet.params)
    nnet.train(feature_set,labels, 100)
    print(nnet.params)
