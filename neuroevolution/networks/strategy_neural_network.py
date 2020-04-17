from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.activation_functions import sigmoid
import numpy as np


class StrategyNeuralNetwork(GeneticNeuralNetwork):
  def __init__(self, layers:list, num_of_classes:int, input_size:int,
               activation_functs = None, pop_size = 50, sigma = 0.1, lr = 0.001):
    self.learning_rate = lr
    self.sigma = sigma
    self.pop_size = pop_size
    self.max_pop_size = pop_size
    self.layers = layers
    self.population = {}
    self.additions = {}
    if self.learning_rate <= 0:
      raise AttributeError("The learning rate parameter can't be 0 or lower")
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
    for i in range(epochs):
      self.mutate_population(sigma=self.sigma)
      # Normalize the mutations
      self.normalize_mutations()
      # Make the selection
      self.population.update(self.additions)
      activated_results = self.evolved_feed_forward(inputs)
      self.calculate_loss(activated_results,targets)
      self.sort_population()
      self.selection_operator(1)

  def normalize_mutations(self):
    for (k,v) in self.additions.items():
      for i in range(len(self.layers)-1):
        if (v['W{}'.format(i+1)]).shape != (1,1):
          v['W{}'.format(i+1)] = (v['W{}'.format(i+1)] - np.mean(v[
              'W{}'.format(i+1)])) / np.std(v['W{}'.format(i+1)])
        if (v['b{}'.format(i+1)]).shape != (1,1) and (
            v['b{}'.format(i+1)]).shape != (1,):
          v['b{}'.format(i+1)] = (v['b{}'.format(i+1)] - np.mean(v[
              'b{}'.format(i+1)])) / np.std(v['b{}'.format(i+1)])
        
  
  
  

