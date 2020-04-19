from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.activation_functions import sigmoid
import numpy as np
import typing


class StrategyNeuralNetwork(GeneticNeuralNetwork):
  """Module that ensembles an evolutionary-strategy based neural network.
  It inherits from genetic neural network since behaviour is similar in a lot of
  ways.
  """

  # I recommended you to use `decimal.Decimal` which is a class that wraps a
  # float and avoid all of his problems. Please try to make the following
  # operation in a python console and record your face:
  # >>> 1.2 - 1.0 # pretends to be shoked.
  def __init__(
      self, layers: typing.List, num_of_classes: int,
      input_size: int, activation_functs = None,
      pop_size: int = 50, sigma: float = 0.1, lr = 0.001
  ):
    """Constructor of the evolutionary strategy based beural network
    
    Arguments:
        layers {list} -- Structure of the network encoded as a list with the 
        number of nodes in each layer. Note: The network built is a fully 
        connected one
        num_of_classes {int} -- The number of classes that is going to 
        distinguish the network.
        input_size {int} -- The number of nodes in the input layer
    
    Keyword Arguments:
        activation_functs {[type]} -- A list of the activation functions used
        in each layer in the forward pass (default: {None})
        pop_size {int} -- the size of the population to be created in order
        to optimize the network (default: {50})
        sigma {float} -- A limitant factor on how much jitter will affect
        the weights (default: {0.1})
        lr {float} -- Learning rate. It determines how fast will the
        algorithm converge (default: {0.001})
    """        
    self.learning_rate = lr
    self.sigma = sigma
    self.pop_size = pop_size
    self.max_pop_size = pop_size
    self.layers = layers
    self.population = {}
    self.additions = {}
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    if pop_size == 0:
      raise AttributeError(
        "Can't create a genetic neural net without a single individual")
    if self.learning_rate <= 0:
      raise AttributeError("The learning rate parameter can't be 0 or lower")
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [sigmoid for i in range(len(self.layers))]
    self.initialize_weithts_and_biases()
  
  def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
    """Training loop, it mutates the weights, biases, selects and orders the
    population in order to reduce the objective function, in this case
    crossentropy
    
    Arguments:
        inputs {np.ndarray} -- Input array with the features
        targets {np.ndarray} -- Target array, or labels
        epochs {int} -- Number of iterations
    """
    for _ in range(epochs):
      self.mutate_population(sigma=self.sigma)
      # Normalize the mutations
      self.normalize_mutations()
      # Make the selection
      self.population.update(self.additions)
      self.additions = {}
      activated_results = self.evolved_feed_forward(inputs)
      self.calculate_loss(activated_results,targets)
      self.sort_population()
      self.selection_operator(1)

  def normalize_mutations(self):
    """Function that normalizes the mutations done to the current population
    """
    for v in self.additions.values():
      for i in range(len(self.layers)-1):
        if (v['W{}'.format(i+1)]).shape != (1,1):
          v['W{}'.format(i+1)] = (v['W{}'.format(i+1)] - np.mean(v[
              'W{}'.format(i+1)])) / np.std(v['W{}'.format(i+1)])
        if (v['b{}'.format(i+1)]).shape != (1,1) and (
            v['b{}'.format(i+1)]).shape != (1,):
          v['b{}'.format(i+1)] = (v['b{}'.format(i+1)] - np.mean(v[
              'b{}'.format(i+1)])) / np.std(v['b{}'.format(i+1)])
        
  
  
  

