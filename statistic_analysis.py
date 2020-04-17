from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.networks.annealed_neural_network import AnnealedNeuralNetwork
from neuroevolution.networks.strategy_neural_network import StrategyNeuralNetwork
from neuroevolution.networks.random_search_nn import RandomSearchNeuralNetwork
import numpy as np
import statistics
basic_loss = []
genetic_loss = []
annealed_loss = []
random_loss = []
epochs = 100

if __name__ == "__main__":
  feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
  labels = np.array([[1,0,0,1,1]])
  labels = labels.reshape(5,1)
  for i in range(30):
    print("Epoch: {}".format(i))
    # Basic Neural Network Execution
    nnet = BasicNeuralNetwork(layers=[3,1], input_size=3, num_of_classes= 1)
    nnet.train(feature_set,labels,epochs)
    basic_loss.append(nnet.loss[-1])
    # Genetic Neural Network Execution
    nnet = GeneticNeuralNetwork(layers=[3,1], input_size=3, num_of_classes= 1)
    nnet.train(feature_set,labels,epochs)
    genetic_loss.append(nnet.population[
          list(nnet.population.keys())[0] ]['loss'])
    # Annealed Neural Network Execution
    nnet = AnnealedNeuralNetwork(layers=[3,1], input_size=3, num_of_classes= 1)
    nnet.train(feature_set,labels,epochs)
    annealed_loss.append(nnet.loss[-1])
    # Random Neural Network Execution
    nnet = RandomSearchNeuralNetwork(layers=[3,1], input_size=3, num_of_classes= 1)
    nnet.train(feature_set,labels,epochs)
    random_loss.append(nnet.loss[-1])

  print("Basic Neural Network: ")
  print("Mean: {}".format(str(statistics.mean(basic_loss))))
  print("Variance: {}".format(str(statistics.variance(basic_loss))))
  print("Std Deviation: {}".format(str(statistics.stdev(basic_loss))))
  print('------')
  print("Genetic Neural Network: ")
  print("Mean: {}".format(str(statistics.mean(genetic_loss))))
  print("Variance: {}".format(str(statistics.variance(genetic_loss))))
  print("Std Deviation: {}".format(str(statistics.stdev(genetic_loss))))
  print('------')
  print("Annealed Neural Network: ")
  print("Mean: {}".format(str(statistics.mean(annealed_loss))))
  print("Variance: {}".format(str(statistics.variance(annealed_loss))))
  print("Std Deviation: {}".format(str(statistics.stdev(annealed_loss))))
  print('------')
  print("Random Search Neural Network: ")
  print("Mean: {}".format(str(statistics.mean(random_loss))))
  print("Variance: {}".format(str(statistics.variance(random_loss))))
  print("Std Deviation: {}".format(str(statistics.stdev(random_loss))))
  print('------')
  