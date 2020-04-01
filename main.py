from basic_neural_network import BasicNeuralNetwork
from neuron import Neuron
import numpy as np

if __name__ == "__main__":
    input_layer = []
    output_layer = []
    for i in range(3):
        input_layer.append(Neuron(3, random=True))
    output_layer.append(Neuron(1, random=True))
    # print(output_layer)
    #Creating the actual nnet
    nnet = BasicNeuralNetwork()
    nnet.add_layer(input_layer,0)
    nnet.add_layer(output_layer,0)
    nnet.train([0,0,1], [1])
