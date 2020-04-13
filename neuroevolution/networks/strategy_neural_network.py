from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.activation_functions import sigmoid
import numpy


class StrategyNeuralNetwork(GeneticNeuralNetwork):
  def __init__(self, layers:list, num_of_classes:int, input_size:int,
               activation_functs = None, pop_size = 50, sigma = 0.1, lr = 0.001):
    self.learning_rate = lr
    self.pop_size = pop_size
    self.sigma = sigma
    self.layers = layers

    self.population = {}
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [sigmoid for i in range(len(self.layers))]
    self.initialize_weithts_and_biases()
  
  def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
    # TODO: Generate samples in a uniform way
    # TODO: Calculate a jittered version of it
    # TODO: Standrize the rewards
    # TODO: Apply them using the learning rate ponderation
    

  

