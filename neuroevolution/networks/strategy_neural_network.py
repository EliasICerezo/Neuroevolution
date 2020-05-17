from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.activation_functions import sigmoid
import numpy as np
import typing

class StrategyNeuralNetwork(GeneticNeuralNetwork):
  """Module that ensembles an evolutionary-strategy based neural network.
  It inherits from genetic neural network since behaviour is similar in a lot of
  ways.
  """
  def __init__(self, layers:list, num_of_classes:int, input_size:int,
               activation_functs = None, pop_size = 50, sigma = 0.1, lr = 0.001):
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
    self.layers = layers
    self.population = {}
    self.additions = {}
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    if pop_size <1:
      raise AttributeError(
        "Can't create a genetic neural net with less than 1 individual")
    if self.learning_rate <= 0:
      raise AttributeError("The learning rate parameter can't be 0 or lower")
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [sigmoid for i in range(len(self.layers))]
    self.initialize_weithts_and_biases(pop_size=1)
  
  def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
    """Training loop, it mutates the weights, biases, selects and orders the
    population in order to reduce the objective function, in this case
    crossentropy
    
    Arguments:
        inputs {np.ndarray} -- Input array with the features
        targets {np.ndarray} -- Target array, or labels
        epochs {int} -- Number of iterations
    """
    if len(self.population.keys()) != 1:
      raise AttributeError("Can't have more than one solution in this state of the program")
    activated_results = self.evolved_feed_forward(inputs)
    self.calculate_loss(activated_results,targets)
    original_solution = self.population[list(self.population.keys())[0]]
    print("Initial loss: {}".format(original_solution['loss']))
    for i in range(epochs):
      self.initialize_weithts_and_biases(pop_size=self.pop_size)
      self.jitter_population(original_solution)
      activated_results = self.evolved_feed_forward(inputs)
      self.calculate_loss(activated_results,targets)
      standard_losses = self.standarize_loss()
      self.update_weights(standard_losses,original_solution)
      self.additions = {}
    
    self.population = {"final_solution": original_solution}
    activated_results = self.evolved_feed_forward(inputs)
    self.calculate_loss(activated_results,targets)

  def update_weights(self, standard_losses:np.array, original_solution:dict):
    for i,v in enumerate(self.population.values()):
      for (a,b) in v.items():
        original_solution[a] = original_solution[a] + self.learning_rate/(
            self.pop_size*self.sigma) * np.dot(b,standard_losses[i])

  def standarize_loss(self):
    vs = list(self.population.values())
    losses = []
    for i in vs:
      losses.append(i['loss'])
    losses = np.asarray(losses)
    return (losses - np.mean(losses)) / np.std(losses)
    

  def jitter_population(self, original_solution:dict):
    # breakpoint()
    for (_,v) in self.population.items():
      jittered_individual = {}
      for (a,b) in v.items():
        # breakpoint()
        jittered_individual[a] = original_solution[a] + (self.sigma * b)
      # breakpoint()
      self.new_individual(jittered_individual)
    self.population = self.additions
