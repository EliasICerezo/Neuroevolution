from neuroevolution.basic_neural_network import BasicNeuralNetwork
from neuroevolution.neuroevolved_neural_network import NeuroEvolvedNeuralNetwork
import numpy as np

if __name__ == "__main__":
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    # print(output_layer)
    #Creating the actual nnet
    nnet = NeuroEvolvedNeuralNetwork(layers=[3,2,1], input_size=3, num_of_classes= 1)
    print(nnet.params)
    print(nnet.population.keys())
    print(nnet.feed_forward)
    nnet.train(feature_set,labels, 100)
    vs = nnet.population.values()
    vs = list(vs)
    for i in vs: print(i['loss'])
    # print(nnet.population)
    # print(nnet.params)
